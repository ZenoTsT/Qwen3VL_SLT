import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration

model_name = "Qwen/Qwen3-VL-4B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(name)