# python scripts/frame_stats.py \
#   --json /Users/zenotesta/Documents/GitHub/Qwen3VL_SLT/data/phoenix_dataset.json \
#   --root_dir /Users/zenotesta/Desktop/Tirocinio/Datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
#   --split train

import os
import json
import glob
import math
import argparse
from statistics import mean, median


def percentile(values, p):
    """
    p in [0, 100]
    Returns the percentile using nearest-rank style on sorted values.
    """
    if not values:
        return None
    values = sorted(values)
    k = math.ceil((p / 100) * len(values)) - 1
    k = max(0, min(k, len(values) - 1))
    return values[k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to dataset json")
    parser.add_argument("--root_dir", default="", help="Optional root dir to prepend to video_path")
    parser.add_argument("--split", default="train", help="Dataset split: train/dev/test")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["splits"][args.split]

    frame_counts = []
    info = []

    for ex in items:
        pattern = ex["video_path"].lstrip("/")

        if args.root_dir:
            pattern = os.path.join(args.root_dir, pattern)

        pattern = os.path.normpath(pattern)

        frames = sorted(glob.glob(pattern))
        n_frames = len(frames)

        if n_frames == 0:
            print(f"[WARNING] no frames found for pattern: {pattern}")
            continue

        frame_counts.append(n_frames)
        info.append({
            "video_path": pattern,
            "n_frames": n_frames,
            "sentence": ex.get("sentence", "")
        })

    if not frame_counts:
        print("No valid samples found.")
        return

    max_item = max(info, key=lambda x: x["n_frames"])

    print(f"\nSplit: {args.split}")
    print(f"Num valid samples: {len(frame_counts)}")

    print("\n--- Max frames sample ---")
    print(f"Frames: {max_item['n_frames']}")
    print(f"Video path: {max_item['video_path']}")
    print(f"Sentence: {max_item['sentence']}")

    print("\n--- Stats ---")
    print(f"Mean frames: {mean(frame_counts):.2f}")
    print(f"Median frames: {median(frame_counts):.2f}")
    print(f"P95 frames: {percentile(frame_counts, 95)}")
    print(f"P98 frames: {percentile(frame_counts, 98)}")
    print(f"P99 frames: {percentile(frame_counts, 99)}")


if __name__ == "__main__":
    main()