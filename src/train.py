import argparse
import torch
import os

from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader

from data import load_split, SLTDataset, make_collate_fn
from qwen_lora import build_processor, build_model_with_lora
from train_loop import train


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json", required=True)
    parser.add_argument("--root_dir", default="")

    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--early_stopping", type=float, default=3)

    # dataloader
    parser.add_argument("--num_workers", type=int, default=2)

    # lora (kept minimal but configurable)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.15)
    parser.add_argument("--lora_scope", type=str, default="joint")

    # logging
    parser.add_argument("--log_every", type=int, default=10)
    
    # FPS sampling
    parser.add_argument("--orig_fps", type=int)
    parser.add_argument("--target_fps", type=int)
    
    # Checkpoints
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--resume", default="")

    return parser.parse_args()


def main():
    args = parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, static_graph=True) # Prova
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ----------------
    # Data
    # ----------------
    train_samples = load_split(args.json, "train", args.root_dir, args.target_fps, args.orig_fps)
    val_samples = load_split(args.json, "dev", args.root_dir, args.target_fps, args.orig_fps)

    train_dataset = SLTDataset(train_samples)
    val_dataset = SLTDataset(val_samples)

    processor, tokenizer = build_processor(args.model)
    
    video_fps = args.target_fps if args.target_fps and args.target_fps > 0 else args.orig_fps
    collate_fn = make_collate_fn(processor, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ----------------
    # Model + LoRA
    # ----------------
    model = build_model_with_lora(
        args.model,
        dtype=torch.bfloat16,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_scope=args.lora_scope,
    )

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Prepare for DDP / mixed precision
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # ----------------
    # Train
    # ----------------
    train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        accelerator=accelerator,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        output_dir=args.output_dir,
        resume_dir=args.resume,
        log_every_updates=args.log_every,
        early_stopping_patience=args.early_stopping,
    )

    if accelerator.is_main_process:
        print("Done.")


if __name__ == "__main__":
    main()