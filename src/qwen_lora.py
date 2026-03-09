import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


def build_processor(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    return processor, tokenizer

def _collect_qwen3vl_lora_targets(model, lora_scope: str):
    """
    Return the FULL module names to target with LoRA.

    lora_scope:
      - "text"   -> only language_model attention/MLP
      - "vision" -> only visual attention/MLP (+ merger)
      - "joint"  -> both
    """
    if lora_scope == "all-linear":
        return "all-linear"
    
    text_suffixes = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    vision_suffixes = (
        "attn.qkv",
        "attn.proj",
        "mlp.linear_fc1",
        "mlp.linear_fc2",
        "merger.linear_fc1",
        "merger.linear_fc2",
    )

    targets = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if lora_scope in ("text", "joint"):
            if name.startswith("language_model.") and name.endswith(text_suffixes):
                targets.append(name)

        if lora_scope in ("vision", "joint"):
            if name.startswith("visual.") and name.endswith(vision_suffixes):
                targets.append(name)

    return sorted(set(targets))


def build_model_with_lora(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_scope: str = "all-linear",  # "text", "vision", "joint", "all-linear"
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
    model.gradient_checkpointing_enable()  
    model.config.use_cache = False       
    
    # 2) Choose target modules
    target_modules = _collect_qwen3vl_lora_targets(model, lora_scope)
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