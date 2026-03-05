# data.py

import os
import glob
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from qwen_vl_utils import process_vision_info


PROMPT_TEXT = "Translate this sign language video to German."


@dataclass
class Sample:
    frame_paths: List[str]
    target: str
    meta: Dict[str, Any]


# ---------------------------
# JSON loading
# ---------------------------

def load_split(json_path: str, split: str, root_dir: str = "", target_fps: int = 0, source_fps: int = 25) -> List[Sample]:

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["splits"][split]

    samples = []

    for ex in items:

        pattern = ex["video_path"].lstrip("/")

        if root_dir:
            pattern = os.path.join(root_dir, pattern)

        pattern = os.path.normpath(pattern)

        frames = sorted(glob.glob(pattern))

        if len(frames) == 0:
            raise RuntimeError(f"No frames found for pattern {pattern}")
        
        if target_fps and target_fps < source_fps:
            stride = max(1, round(source_fps / target_fps)) 
            frames = frames[::stride]

        samples.append(
            Sample(
                frame_paths=frames,
                target=ex["sentence"],
                meta=ex
            )
        )

    return samples


# ---------------------------
# Dataset
# ---------------------------

class SLTDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):

        s = self.samples[i]

        return {
            "frame_paths": s.frame_paths,
            "target": s.target,
        }


# ---------------------------
# Collate Function
# ---------------------------

# Check https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl, paragraphs Qwen3VLModel and Qwen3VLForConditionalGeneration
# Qwen can manage by himself "attention_mask" and "labels"
# attention_mask: “Mask to avoid performing attention on padding token indices… 1 = non masked, 0 = masked” 
# labels: “Labels for computing the masked language modeling loss… Tokens set to -100 are ignored (masked)”

def make_collate_fn(processor, tokenizer):

    # Initialize (if exists) the padding
    if tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = tokenizer.eos_token_id

    def collate(batch):
        
        DEBUG = os.getenv("DEBUG_COLLATE", "0") == "1"
        if DEBUG:
            print(f"\n[collate] batch_size={len(batch)}")
            print(f"[collate] pad_id={pad_id}, eos_id={tokenizer.eos_token_id}, pad_token_id={tokenizer.pad_token_id}")
            for bi, ex in enumerate(batch):
                print(f"[collate] sample[{bi}] #frames={len(ex['frame_paths'])}, target_len_chars={len(ex['target'])}")

        messages_list = []
        texts = []

        for ex in batch:

            video_uris = []

            for p in ex["frame_paths"]:
                ap = os.path.abspath(p)             # I take the abs path of every frame
                video_uris.append("file://" + ap)   # I save them in video_uris

            # I Build the messages for Qwen
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_uris,
                        },
                        {
                            "type": "text",
                            "text": PROMPT_TEXT,
                        },
                    ],
                }
            ]
            messages_list.append(messages)

            # Using the procesor of Qwen, I convert the message into an input Qwen-friendly
            text = processor.apply_chat_template(
                messages,
                tokenize=False,                         # I don't tokenize the input yet
                add_generation_prompt=True              # I add the token to let the model generate
            )
            texts.append(text)
            
            if DEBUG and len(texts) == 1:
                print("\n[collate] chat_template example (first sample):")
                print(texts[0][:600] + ("..." if len(texts[0]) > 600 else ""))
            
        # Now I have:
        # messages_list -> list of Qwen messages
        # texts -> list of Qwen inputs

        image_inputs_batch = []
        video_inputs_batch = []

        for messages in messages_list:
            
            # process_vision_info takes the paths in the fields "video" and "image" of a message
            # and convert them into PIL images (in this case image_inputs will be None)
            image_inputs, video_inputs = process_vision_info(messages) 
            video_inputs_batch.append(video_inputs)
            
            if DEBUG and len(video_inputs_batch) == 1:
                try:
                    n_vid = len(video_inputs) if video_inputs is not None else 0
                    n_frames = len(video_inputs[0]) if (video_inputs and len(video_inputs) > 0) else 0
                    print(f"\n[collate] process_vision_info: n_videos={n_vid}, n_frames_in_first_video={n_frames}")
                except Exception as e:
                    print(f"\n[collate] process_vision_info: could not introspect structure: {type(video_inputs)} err={e}")
                    
        # The Qwen processor converts:
        # - the prompt strings (texts) into input_ids + attention_mask
        # - the videos into vision tensors (pixel_values_videos + video_grid_thw, etc.)
        #
        # padding=True is REQUIRED to make a batch tensor for input_ids (otherwise lengths differ).
        # Here, the processor pads ONLY the prompt part (not our target).
        proc = processor(
            text=texts,
            videos=video_inputs_batch,
            padding=True,
            return_tensors="pt",
        )
        
        if DEBUG:
            print("\n[collate] proc keys:", list(proc.keys()))
            for k, v in proc.items():
                if hasattr(v, "shape"):
                    print(f"[collate]   {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    print(f"[collate]   {k}: type={type(v)}")
            # test prompt length (padded) vs true prompt length for sample
            print(f"[collate] input_ids.shape={tuple(proc['input_ids'].shape)} attention_mask.shape={tuple(proc['attention_mask'].shape)}")

        input_ids = proc["input_ids"]                # (B, Pmax)
        attention_mask = proc["attention_mask"]      # (B, Pmax) 1=real token, 0=prompt padding

        B, Pmax = input_ids.shape

        # 1) Tokenize ALL targets and store them as 1D tensors (one per sample)
        tgt_ids_list = []
        tgt_lens = []
        for ex in batch:
            tgt_ids = tokenizer(
                ex["target"],
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"][0]  # [0] because tokenizer returns a batch dimension (1, L)

            # Add EOS at end of the target
            eos = torch.tensor([tokenizer.eos_token_id], dtype=tgt_ids.dtype)
            tgt_ids = torch.cat([tgt_ids, eos], dim=0)

            tgt_ids_list.append(tgt_ids)
            tgt_lens.append(int(tgt_ids.numel()))

        Tmax = max(tgt_lens) if len(tgt_lens) > 0 else 0
        final_len = Pmax + Tmax

        # 2) Allocate the final tensors for the whole batch
        input_ids2 = torch.full((B, final_len), pad_id, dtype=input_ids.dtype)  # Create the final tensor that will contain the whole sequence (all padding for now)
        attn2 = torch.zeros((B, final_len), dtype=attention_mask.dtype)         # Create the final tensor that will contain the whole attention mask (all 0 for now)
        labels2 = torch.full((B, final_len), -100, dtype=input_ids.dtype)       # Create the final tensor that will contain the whole masking + groundh truth

        # Copy prompt tensors (already padded) into the first part
        input_ids2[:, :Pmax] = input_ids    # for each row [:, X], i modify all the column until Pmax [X, :Pmax] (I insert the prompt)
        attn2[:, :Pmax] = attention_mask    # same with the attention_mask

        # 3) Copy each target into the tail (after prompt)
        for i, tgt_ids in enumerate(tgt_ids_list):
            L = int(tgt_ids.numel())                # lenght of the ground truth
            input_ids2[i, Pmax:Pmax + L] = tgt_ids  # I first put the ground truth after the prompt + padding (of the prompt)
            attn2[i, Pmax:Pmax + L] = 1             # I put 1 in attention mask
            labels2[i, Pmax:Pmax + L] = tgt_ids     # I don't put -100 in these position (ground truth uncovered)

            if DEBUG and i == 0:
                # prompt_len is the "true" prompt length (not counting prompt padding)
                prompt_len = int(attention_mask[i].sum().item())

                print(f"\n[collate] sample0 prompt_len(true)={prompt_len} (Pmax={Pmax})")
                print(f"[collate] sample0 tgt_len_tokens(with eos)={L} (Tmax={Tmax})")
                print(f"[collate] sample0 final_len={final_len}")

                num_ignored = int((labels2[i] == -100).sum().item())
                num_supervised = int((labels2[i] != -100).sum().item())

                active = (labels2[i] != -100).nonzero(as_tuple=True)[0]

                print(f"[collate] supervised span start={active.min().item()} end={active.max().item()}")
                print(f"[collate] supervised tokens count={active.numel()}")
                print(f"[collate] expected target len={L}")
                print(f"[collate] Pmax={Pmax}")
                print(f"[collate] sample0 labels: ignored={num_ignored} supervised={num_supervised}")

        # Replace text tensors in proc; keep vision tensors from proc
        proc["input_ids"] = input_ids2
        proc["attention_mask"] = attn2
        proc["labels"] = labels2

        if DEBUG:
            print("\n[collate] after fixed padding (single pass):")
            print(f"[collate] input_ids2={tuple(input_ids2.shape)} attn2={tuple(attn2.shape)} labels2={tuple(labels2.shape)}")

            # sanity checks:
            # - pad positions in attention mask should correspond to either prompt padding or target padding
            pad_positions = (attn2 == 0).sum().item()
            ignored_on_pad = ((attn2 == 0) & (labels2 == -100)).sum().item()
            print(f"[collate] pad_positions={int(pad_positions)}, pad_positions_with_labels_-100={int(ignored_on_pad)}")

        return proc

    return collate