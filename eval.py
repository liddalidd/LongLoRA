# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from peft import PeftModel
from llama_attn_replace import replace_llama_attn
import json
import re


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=2048, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=True, help='')
    parser.add_argument('--data_path', type=str, default="./test.bin", help='')
    args = parser.parse_args()
    return args

def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256, resume_ckpt=None):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()
    if resume_ckpt == None:
        resume_ckpt = 0
    else:
        resume_ckpt += batch_size
    for idx in range(resume_ckpt, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y

def val_step_num(data, batch_size, seq_length, sliding_window, resume_ckpt):
    temp = np.ceil((len(data) - seq_length) / sliding_window) - 1
    resume_ckpt =  resume_ckpt + batch_size if resume_ckpt is not None else 0
    return int(np.ceil((temp - resume_ckpt) / batch_size) + 1)


def evaluate(model, data, batch_size, device, seq_length, sliding_window=256, use_cache=False, save_steps=2000):
    stats = {}

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []
    resume_ckpt = None
    resume_idx = None
    # load from ckpt
    if os.path.exists(os.path.join(args.peft_model, f"eval_ckpt_{args.seq_len}_{args.context_size}")):
        eval_path = os.path.join(args.peft_model, f"eval_ckpt_{args.seq_len}_{args.context_size}")
        eval_list = os.listdir(eval_path)
        eval_list.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        resume_ckpt = eval_list[-1]
        print(f'Resume from eval ckpt:{resume_ckpt}')
        resume_idx = int(re.findall(r'\d+', resume_ckpt)[0])
        loss_step_list_val = np.load(os.path.join(eval_path, resume_ckpt, "loss_step_list_val.npy"), allow_pickle=True).tolist()
        loss_list_val = np.load(os.path.join(eval_path, resume_ckpt, "loss_list_val.npy"), allow_pickle=True).tolist()
        acc_list = np.load(os.path.join(eval_path, resume_ckpt, "acc_list.npy"), allow_pickle=True).tolist()

    with torch.no_grad():
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)
        for idx, (x, y) in tqdm(
            enumerate(
                get_as_batch(
                    data['val'], 
                    seq_length, 
                    batch_size, 
                    device=device,
                    sliding_window=sliding_window,
                    resume_ckpt=resume_idx
                )
            ),
            total=val_step_num(data['val'], batch_size, seq_length, sliding_window, resume_idx)
        ):
            idx = idx + resume_idx if resume_idx is not None else idx
            # print(idx)
            val_loss = 0.
            acc = 0.
            cnt = 0

            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                part_len = x[:, i:i + seq_length].shape[1]

                outputs = model(
                    input_ids=x[:, i:i + seq_length],
                    labels=x[:, i:i+seq_length].contiguous(),
                    use_cache=use_cache)

                val_loss = outputs.loss * part_len + val_loss
                acc = ((outputs.logits.argmax(-1) == y[:, i:i+seq_length]).float().sum()) + acc
                cnt += part_len
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(outputs.loss.item())
            val_loss /= cnt
            acc /= cnt
            
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())
            # print(idx, resume_ckpt)
            if idx % 200 == 0 and ((idx != resume_idx and resume_idx != None) or (idx != 0 and resume_idx == None)):
                eval_path = os.path.join(args.peft_model, f"eval_ckpt_{args.seq_len}_{args.context_size}")
                if not os.path.exists(eval_path):
                    os.mkdir(eval_path)
                os.mkdir(os.path.join(eval_path, f"eval_{idx}"))
                np.save(os.path.join(eval_path, f"eval_{idx}", "loss_step_list_val.npy"), loss_step_list_val)
                np.save(os.path.join(eval_path, f"eval_{idx}", "loss_list_val.npy"), loss_list_val)
                np.save(os.path.join(eval_path, f"eval_{idx}", "acc_list.npy"), acc_list)
                print(f'eval ckpt "eval_{idx}" saved ')

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1))

    return stats

def main(args):

    device = "cuda:0"
    seed = 2
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data = {'val': np.memmap(args.data_path, dtype=np.uint16, mode='r')}

    print(f"Num validation tokens: {len(data['val'])}")
    print("data path", args.data_path)
    print("base model", args.base_model)
    print("peft model", args.peft_model)

    if args.flash_attn:
        replace_llama_attn(use_flash_attn=True, use_full=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    if args.peft_model:
        trainable_params = os.path.join(args.peft_model, "trainable_params.bin")
        if os.path.isfile(trainable_params):
            model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
        else:
            raise ValueError("Trainable input embedding and normalization are required.")
        model = PeftModel.from_pretrained(
            model,
            args.peft_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    stats = evaluate(model, data, args.batch_size, device, args.seq_len, sliding_window=256)

    print(stats)
    json.dump(stats, open(os.path.join(args.peft_model,f"eval_stats_{args.seq_len}_{args.context_size}.json"), "w"))


if __name__ == "__main__":
    args = parse_config()
    main(args)
