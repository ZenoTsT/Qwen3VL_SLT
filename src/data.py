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

def load_split(json_path: str, split: str, root_dir: str = "") -> List[Sample]:

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

        # The Qwen processor converts input images and texts in tensors and tokens
        # In this way we can give the input batch to Qwen 
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

        input_ids = proc["input_ids"]
        attention_mask = proc["attention_mask"] # Tells which token are paddings (right now, no one)

        new_input_ids = []
        new_attention = []
        new_labels = []

        for i, ex in enumerate(batch):

            prompt_len = int(attention_mask[i].sum().item())    # Calculate where prompt finish
            prompt_ids = input_ids[i, :prompt_len]              # I take only the prompt

            # I take the target ground truth and I convert it into tokens
            tgt_ids = tokenizer(
                ex["target"],
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"][0]       # I take the tensor (input_ids field of the tokenizer)

            eos = torch.tensor([tokenizer.eos_token_id], dtype=tgt_ids.dtype)   # I take a End Of Sequence token
            tgt_ids = torch.cat([tgt_ids, eos], dim=0)      # I concat it to my ground truth
            ids = torch.cat([prompt_ids, tgt_ids], dim=0)   # And I concat everything in a single prompt (Text tokens + visual placeholders (prompt_ids) + Ground truth)

            labels = torch.cat(
                [torch.full_like(prompt_ids, -100), tgt_ids],   # full_like create a vector filled with -100 with the lenght of prompt_ids
                dim=0
            )
            
            if DEBUG and i == 0:
                print(f"\n[collate] sample0 prompt_len={prompt_len}")
                print(f"[collate] sample0 tgt_len_tokens(with eos)={tgt_ids.numel()}")
                print(f"[collate] sample0 ids_len={ids.numel()} labels_len={labels.numel()}")
                # check labels masking: how many -100 and how many supervised token 
                num_ignored = int((labels == -100).sum().item())
                num_supervised = int((labels != -100).sum().item())
                print(f"[collate] sample0 labels: ignored={num_ignored} supervised={num_supervised}")

            attn = torch.ones_like(ids) # The attention mask is all 1 since we have not added the padding yet

            new_input_ids.append(ids)   # The full prompts
            new_attention.append(attn)  # All the attention masks (tell that until here we don't have padding)
            new_labels.append(labels)   # The vector filled with -100 to not calculate the loss on prompt tokens

        max_len = max(x.size(0) for x in new_input_ids) # I take the longhest prompt

        # Function to apply padding
        def pad_1d(x, value):

            if x.size(0) == max_len:
                return x

            pad = torch.full((max_len - x.size(0),), value, dtype=x.dtype)

            return torch.cat([x, pad], dim=0)  # I apply the padding

        # I apply the padding to the full prompt
        input_ids2 = torch.stack(
            [pad_1d(x, pad_id) for x in new_input_ids],
            dim=0
        )

        # I apply the padding to the att mask (tell that until here we don't have padding)
        attn2 = torch.stack(
            [pad_1d(x, 0) for x in new_attention],
            dim=0
        )

        # I apply the padding to the prompts masking
        labels2 = torch.stack(
            [pad_1d(x, -100) for x in new_labels],
            dim=0
        )

        proc["input_ids"] = input_ids2
        proc["attention_mask"] = attn2
        proc["labels"] = labels2
        
        if DEBUG:
            print("\n[collate] after manual padding:")
            print(f"[collate] input_ids2={tuple(input_ids2.shape)} attn2={tuple(attn2.shape)} labels2={tuple(labels2.shape)}")
            # sanity: attention_mask must be 0 where input_ids is pad (not always perfect, but should match your padding)
            # and labels should be -100 where attn is 0
            pad_positions = (attn2 == 0).sum().item()
            ignored_on_pad = ((attn2 == 0) & (labels2 == -100)).sum().item()
            print(f"[collate] pad_positions={int(pad_positions)}, pad_positions_with_labels_-100={int(ignored_on_pad)}")

        return proc

    return collate