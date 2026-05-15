"""
pipeline/dataset.py — PyTorch Dataset for Amharic language modeling
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict
from pathlib import Path


class AmharicDataset(Dataset):
    """
    Language modeling dataset.
    Returns (input_ids, attention_mask, labels) for causal LM training.
    Labels = input_ids shifted by 1 (predict next token).
    """

    def __init__(
        self,
        texts:      List[str],
        tokenizer,
        max_length: int = 512,
        stride:     int = 256,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.stride     = stride
        self.examples:  List[Dict] = []
        self._build(texts)

    def _build(self, texts: List[str]):
        for text in texts:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            for start in range(0, max(1, len(ids) - self.max_length + 1), self.stride):
                chunk   = ids[start : start + self.max_length]
                if len(chunk) < 16:
                    continue
                pad_len = self.max_length - len(chunk)
                mask    = [1] * len(chunk) + [0] * pad_len
                chunk   = chunk + [self.tokenizer.pad_token_id] * pad_len
                self.examples.append({"input_ids": chunk, "attention_mask": mask})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex     = self.examples[idx]
        ids    = torch.tensor(ex["input_ids"],     dtype=torch.long)
        mask   = torch.tensor(ex["attention_mask"], dtype=torch.long)
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}
