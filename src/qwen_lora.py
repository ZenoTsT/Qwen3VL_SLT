import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


def build_processor(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    return processor, tokenizer

import torch.nn as nn


def collect_qwen3vl_lora_targets(model, scope: str):
    """
    Collect LoRA target modules for Qwen3-VL based on REAL module names.

    scope:
        - "text"   -> only language model linear layers
        - "vision" -> only visual branch linear layers
        - "joint"  -> both text + vision

    Returns:
        list[str] of full module names
    """

    if scope not in {"text", "vision", "joint"}:
        raise ValueError(f"Invalid scope={scope!r}. Use 'text', 'vision', or 'joint'.")

    targets = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Exclude lm_head explicitly
        if name == "lm_head":
            continue

        # -------------------------
        # TEXT BRANCH
        # -------------------------
        if scope in {"text", "joint"}:
            if name.startswith("model.language_model.layers.") and name.endswith((
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )):
                targets.append(name)

        # -------------------------
        # VISION BRANCH
        # -------------------------
        if scope in {"vision", "joint"}:
            if name.startswith("model.visual.blocks.") and name.endswith((
                "attn.qkv",
                "attn.proj",
                "mlp.linear_fc1",
                "mlp.linear_fc2",
            )):
                targets.append(name)

            elif name.startswith("model.visual.merger.") and name.endswith((
                "linear_fc1",
                "linear_fc2",
            )):
                targets.append(name)

            elif name.startswith("model.visual.deepstack_merger_list.") and name.endswith((
                "linear_fc1",
                "linear_fc2",
            )):
                targets.append(name)

    targets = sorted(set(targets))

    if len(targets) == 0:
        raise ValueError(f"No LoRA target modules found for scope={scope!r}")

    return targets


def build_model_with_lora(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_scope: str = "joint",  # "text", "vision", "joint"
):

    #Loads Qwen3-VL and applies LoRA (trainable adapters).
    #Returns a PEFT-wrapped model.

    
    # https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-1M-GGUF/blame/d21e3d7295cbb0717c92997e10d1c069cb12969d/README.md
    # "We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios."
    
    # 1) Load Qwen
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()   # Prova
    model.enable_input_require_grads()      # Prova
    model.config.use_cache = False          # Prova 
    
    # 2) Choose target modules
    target_modules = collect_qwen3vl_lora_targets(model, lora_scope)
    if len(target_modules) == 0:
        raise ValueError(f"No LoRA target modules found for lora_scope={lora_scope!r}")
    print(f"[lora] scope={lora_scope} | num target modules={len(target_modules)}")
    for x in target_modules[:15]:
        print("  ", x)
    if len(target_modules) > 15:
        print("  ...")

    # 3) Apply LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)

    return model