# python scripts/test_collate_fn.py \
#   --model Qwen/Qwen3-VL-2B-Instruct \
#   --json /Users/zenotesta/Documents/GitHub/Qwen3VL_SLT/data/phoenix_dataset.json \
#   --split train \
#   --n 2 \
#   --root_dir /Users/zenotesta/Desktop/Tirocinio/Datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
#   --orig_fps 25 \
#   --target_fps 12

import os
import sys
import argparse
import torch

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# importa la tua load_split + collate
from src.data import load_split, make_collate_fn


def build_batch_from_json(json_path, split, n, root_dir="", target_fps=0, orig_fps=25):
    # Usa la tua load_split reale, così il test riflette davvero il training
    samples = load_split(
        json_path=json_path,
        split=split,
        root_dir=root_dir,
        target_fps=target_fps,
        source_fps=orig_fps,
    )

    samples = samples[:n]

    batch = []
    for s in samples:
        batch.append({
            "frame_paths": s.frame_paths,
            "target": s.target,
        })

    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--json", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1, help="how many samples to test in the batch")
    ap.add_argument("--root_dir", default="")
    ap.add_argument("--orig_fps", type=int, default=25)
    ap.add_argument("--target_fps", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # attiva debug print nella collate
    os.environ["DEBUG_COLLATE"] = "1"

    print("[test] loading processor...")
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    print("[test] loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device)
    model.eval()

    print("[test] building batch from json...")
    batch = build_batch_from_json(
        json_path=args.json,
        split=args.split,
        n=args.n,
        root_dir=args.root_dir,
        target_fps=args.target_fps,
        orig_fps=args.orig_fps,
    )

    video_fps = args.target_fps if args.target_fps and args.target_fps > 0 else args.orig_fps
    print(f"[test] video_fps passed to collate: {video_fps}")

    print("[test] running collate...")
    collate = make_collate_fn(processor, tokenizer, video_fps)
    proc = collate(batch)

    # sposta su device
    for k, v in list(proc.items()):
        if torch.is_tensor(v):
            proc[k] = v.to(args.device)

    # sanity checks
    assert "input_ids" in proc and "attention_mask" in proc and "labels" in proc, "missing core keys"
    assert proc["input_ids"].shape == proc["attention_mask"].shape == proc["labels"].shape, "shape mismatch"
    assert torch.isfinite(proc["input_ids"].float()).all(), "non-finite in input_ids"
    assert torch.isfinite(proc["attention_mask"].float()).all(), "non-finite in attention_mask"

    # labels can contain -100; check others are finite
    valid = proc["labels"] != -100
    if valid.any():
        assert torch.isfinite(proc["labels"][valid].float()).all(), "non-finite in labels valid positions"

    print("[test] forward pass (loss)...")
    with torch.no_grad():
        out = model(**proc)

    print("[test] loss:", float(out.loss.detach().cpu()))
    print("[test] logits shape:", tuple(out.logits.shape))
    print("[test] OK")


if __name__ == "__main__":
    main()