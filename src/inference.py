import os
from typing import List, Tuple

import torch
from tqdm import tqdm
from accelerate import Accelerator

from qwen_vl_utils import process_vision_info
from data import PROMPT_TEXT
from metrics import compute_all


def try_load_adapter_for_inference(model, accelerator: Accelerator, ckpt_dir: str):

    # We unwrap just to be safe with accelerate.prepare / DDP wrappers
    unwrapped = accelerator.unwrap_model(model)

    # PEFT-style adapter loading
    try:
        unwrapped.load_adapter(ckpt_dir, adapter_name="default", is_trainable=False)
        unwrapped.set_adapter("default")
    except Exception as e:
        raise RuntimeError(f"Could not load adapter from {ckpt_dir}. Last error: {e}")

    if accelerator.is_main_process:
        print(f"[ckpt] loaded adapter from: {ckpt_dir}")


@torch.no_grad()
def generate_one(
    model,
    processor,
    tokenizer,
    frame_paths: List[str],
    accelerator: Accelerator,
    max_new_tokens: int = 64,
    num_beams: int = 1,
) -> str:

    # Build video URIs
    video_uris = []
    for p in frame_paths:
        ap = os.path.abspath(p)
        video_uris.append("file://" + ap)

    # Build Qwen messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_uris},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    # Prompt string
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Vision processing
    image_inputs, video_inputs = process_vision_info(messages)

    # Processor -> tensors (single sample)
    proc = processor(
        text=[text],
        videos=[video_inputs],
        padding=True,
        return_tensors="pt",
    )

    # Move to accelerator device
    for k, v in proc.items():
        if torch.is_tensor(v):
            proc[k] = v.to(accelerator.device)

    # Prompt length for slicing the generation from the prompt
    prompt_len = int(proc["attention_mask"][0].sum().item())

    # Generate
    gen_ids = model.generate(
        **proc,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        use_cache=True,
    )

    # Decode only the generation
    cont_ids = gen_ids[0, prompt_len:]
    pred = tokenizer.decode(cont_ids, skip_special_tokens=True).strip()

    return pred


def inference(
    model,
    processor,
    tokenizer,
    test_dataset,
    accelerator: Accelerator,
    ckpt_dir: str,
    max_new_tokens: int = 128,
    num_beams: int = 1,
    limit: int = 0,
) -> Tuple[List[str], List[str], dict]:
    
    # 1) Load adapter
    accelerator.wait_for_everyone()

    if ckpt_dir and os.path.isdir(ckpt_dir):
        try_load_adapter_for_inference(model, accelerator, ckpt_dir)
    else:
        if accelerator.is_main_process:
            print("[ckpt] no checkpoint provided → using base model + empty LoRA")

    accelerator.wait_for_everyone()

    # 2) Inference loop
    model.eval()

    N = len(test_dataset)
    if limit and limit > 0:
        N = min(N, limit)

    preds = []
    refs = []

    pbar = tqdm(
        total=N,
        disable=not accelerator.is_main_process,
        desc="test",
        dynamic_ncols=True,
    )

    for i in range(N):
        ex = test_dataset[i]

        pred = generate_one(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            frame_paths=ex["frame_paths"],
            accelerator=accelerator,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        preds.append(pred)
        refs.append(ex["target"])
        
        if accelerator.is_main_process and i < 5:
            print("\n--- Example", i, "---")
            print("GT  :", ex["target"])
            print("PRED:", pred)

        if accelerator.is_main_process:
            pbar.update(1)

    if accelerator.is_main_process:
        pbar.close()

    # 3) Metrics (compute only on main process)
    metrics = {}
    if accelerator.is_main_process:
        metrics = compute_all(preds, refs)

        print("\n[test] METRICS")
        for k, v in metrics.items():
            print(f"[test] {k}: {v:.6f}")

        print(f"[test] num_samples={len(preds)}")

    accelerator.wait_for_everyone()
    return preds, refs, metrics