import argparse
import torch

from accelerate import Accelerator
from torch.utils.data import DataLoader

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from peft import LoraConfig, get_peft_model, TaskType

from data import load_split, SLTDataset, make_collate_fn


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--json", required=True)
    parser.add_argument("--root_dir", default="")
    parser.add_argument("--split", default="train")

    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--fps", type=float, default=24)

    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="bf16")

    samples = load_split(args.json, args.split, args.root_dir)

    dataset = SLTDataset(samples)

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    collate_fn = make_collate_fn(processor, tokenizer, args.fps)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model, optimizer, dataloader = accelerator.prepare(
        model,
        optimizer,
        dataloader
    )

    model.train()

    optimizer.zero_grad()

    for epoch in range(args.epochs):

        for step, batch in enumerate(dataloader):

            outputs = model(**batch)

            loss = outputs.loss

            accelerator.backward(loss)

            if (step + 1) % args.grad_accum == 0:

                optimizer.step()
                optimizer.zero_grad()


if __name__ == "__main__":
    main()