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
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence, List

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from llama_attn_replace import replace_llama_attn
from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier

from transformers.trainer_utils import is_main_process
import torch.distributed as dist
from loguru import logger



from datasets import load_dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    # LD ADD
    train_data: str = field(
        default="togethercomputer/RedPajama-Data-1T-Sample",
        metadata={"help": "Training data path"},
    )
    local_training_path: str = field(
        default="togethercomputer/RedPajama-Data-1T-Sample",
        metadata={"help": "local training data path"},
    )
    full_window_idx: int = field(
        default=-1,
        metadata={"help": "Position of full attention"},
    )
    window_size: int = field(
        default=256,
        metadata={"help": "window attention size"},
    )


def print_fn(info):
    rank = int(os.environ.get('RANK', -1))
    if rank == 0:
        logger.info(info)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}

def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    training_args.disable_tqdm = True

    # NOTE: May expand supported model types in the future
    # if model_args.model_type == "gpt-neox":
    #     replace_gpt_neox_attn(training_args.use_flash_attn, training_args.use_full_attn)
    # else:
    #     assert model_args.model_type == "llama", "Only support llama and gpt-neox for now"
    #     replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.window_size = training_args.window_size
    config.full_window_idx = training_args.full_window_idx
        
    window_size_list = [config.window_size for _ in range(config.num_hidden_layers)]
    window_size_list[config.full_window_idx] = -1
    config.window_size_list = window_size_list
    print_fn(config)

    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}

    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    config._attn_implementation = "flash_attention_2"

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    model.to("cuda")
    print_fn(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    print_fn(tokenizer)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()
        
    # online process
    # dataset = load_dataset(training_args.train_data, cache_dir=training_args.cache_dir)
    # dataset = dataset.map(partial(tokenize_fn, tokenizer),batched=True, num_proc=128, remove_columns=["text", "meta"])
    # dataset.save_to_disk(some/path)
        
    # off-line load    
    len_version = training_args.model_max_length // 1024
    encoded_path = f"{training_args.local_training_path}/{len_version}k"
    print_fn(f"loading dataset from {encoded_path}...")
    dataset = load_dataset(encoded_path)

    if rank == 0:
        barrier()

    print_fn(dataset)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if training_args.low_rank_training:
        if model_args.model_type == "gpt-neox":
            # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
            targets = ["query_key_value", "dense"]
        else:
            targets=["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        data_collator=data_collator)
    
    train_result = trainer.train(resume_from_checkpoint=True)
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
