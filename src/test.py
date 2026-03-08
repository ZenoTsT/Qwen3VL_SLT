# Se voglio testare in locale:
# python src/test.py \
#   --model Qwen/Qwen3-VL-2B-Instruct \
#   --json /Users/zenotesta/Documents/GitHub/Qwen3VL_SLT/data/phoenix_dataset.json \
#   --root_dir /Users/zenotesta/Desktop/Tirocinio/Datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
#   --ckpt_dir "" \
#   --orig_fps 25 \
#   --target_fps 12 \
#   --max_new_tokens 128 \
#   --num_beams 1 \
#   --limit 1

import argparse
import os
import torch

from accelerate import Accelerator

from data import load_split, SLTDataset
from qwen_lora import build_processor, build_model_with_lora
from inference import inference


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--json", required=True)
    p.add_argument("--root_dir", default="")

    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")

    # fps sampling
    p.add_argument("--orig_fps", type=int, default=25)
    p.add_argument("--target_fps", type=int, default=0)

    # lora (must match training config, or at least be compatible)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # checkpoint dir (the folder containing adapter weights)
    p.add_argument("--ckpt_dir", required=True)

    # generation
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_beams", type=int, default=1)

    # debug
    p.add_argument("--limit", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator(mixed_precision="bf16")

    if accelerator.is_main_process:
        print("[test] START")
        print(f"[test] ckpt_dir={args.ckpt_dir}")
        print(f"[test] root_dir={args.root_dir}")
        print(f"[test] json={args.json}")
        print(f"[test] target_fps={args.target_fps} orig_fps={args.orig_fps}")

    # ----------------
    # Data
    # ----------------
    test_samples = load_split(args.json, "test", args.root_dir, args.target_fps, args.orig_fps)

    test_dataset = SLTDataset(test_samples)

    # ----------------
    # Model + LoRA
    # ----------------
    processor, tokenizer = build_processor(args.model)

    model = build_model_with_lora(
        args.model,
        dtype=torch.bfloat16,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Prepare (even if single GPU, this keeps the same style)
    model = accelerator.prepare(model)

    # ----------------
    # Inference 
    # ----------------
    inference(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        accelerator=accelerator,
        ckpt_dir=args.ckpt_dir,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        limit=args.limit,
    )

    if accelerator.is_main_process:
        print("[test] END")


if __name__ == "__main__":
    main()