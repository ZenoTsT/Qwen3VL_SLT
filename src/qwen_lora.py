import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


def build_processor(model_name: str):
    """
    Loads the Qwen processor and returns (processor, tokenizer).
    """
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    return processor, tokenizer


def build_model_with_lora(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """
    Loads Qwen3-VL and applies LoRA (trainable adapters).
    Returns a PEFT-wrapped model.
    """
    
    # https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-1M-GGUF/blame/d21e3d7295cbb0717c92997e10d1c069cb12969d/README.md
    # "We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios."
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()   # Giusto per prova 
    model.config.use_cache = False          # Giusto per prova 

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    model = get_peft_model(model, lora_config)

    return model