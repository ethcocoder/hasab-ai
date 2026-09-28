from .dataset import AmharicDataset
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
        stride:     int = 256,     # Overlapping windows for better coverage
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.stride     = stride
        self.examples:  List[Dict] = []
        self._build(texts)

    def _build(self, texts: List[str]):
        for text in texts:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            # Sliding window over long texts
            for start in range(0, max(1, len(ids) - self.max_length + 1), self.stride):
                chunk = ids[start : start + self.max_length]
                if len(chunk) < 16:
                    continue
                pad_len = self.max_length - len(chunk)
                mask    = [1] * len(chunk) + [0] * pad_len
                chunk   = chunk + [self.tokenizer.pad_token_id] * pad_len
                self.examples.append({
                    "input_ids":      chunk,
                    "attention_mask": mask,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids  = torch.tensor(ex["input_ids"],      dtype=torch.long)
        mask = torch.tensor(ex["attention_mask"],  dtype=torch.long)
        # Labels: -100 for padding (ignored in loss)
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


"""
pipeline/finetune.py — Training loop with LCE joint training
"""

import torch
import time
import json
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional


class Trainer:
    """
    Professional training loop for AmharicGPT2 with:
      - Gradient accumulation
      - Mixed precision (fp16)
      - LCE joint training warmup
      - Checkpointing
      - W&B logging (optional)
    """

    def __init__(self, model, tokenizer, config, device: Optional[str] = None):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = config
        configured_device = getattr(config.training, "device", "auto")
        if device:
            self.device = device
        elif configured_device == "cuda":
            self.device = "cuda"
        elif configured_device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.history   = []
        print(f"🔧 Training on: {self.device}")

    def train(self, train_dataset, val_dataset=None):
        tc = self.cfg.training

        train_loader = DataLoader(
            train_dataset,
            batch_size  = tc.batch_size,
            shuffle     = True,
            num_workers = tc.num_workers,
            pin_memory  = (self.device == "cuda"),
        )

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr           = tc.learning_rate,
            weight_decay = tc.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=tc.max_steps)
        if tc.fp16 and self.device == "cuda":
            try:
                scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        # ── Phase 1: Freeze GPT-2, train LCE only ────────────────────────────
        print("\n📍 Phase 1: Training LCE bottleneck (GPT-2 frozen)")
        self.model.freeze_gpt2()
        global_step = self._run_loop(
            train_loader, optimizer, scheduler, scaler,
            max_steps=tc.lce_warmup_steps, phase="LCE_warmup",
            val_dataset=val_dataset,
        )

        # ── Phase 2: Unfreeze all, full fine-tuning ───────────────────────────
        print("\n📍 Phase 2: Full fine-tuning (all weights)")
        self.model.unfreeze_all()
        self._run_loop(
            train_loader, optimizer, scheduler, scaler,
            max_steps=tc.max_steps, start_step=global_step, phase="full_finetune",
            val_dataset=val_dataset,
        )

        print("\n✅ Training complete!")
        return self.history

    def _run_loop(
        self, loader, optimizer, scheduler, scaler,
        max_steps, start_step=0, phase="train", val_dataset=None
    ):
        tc    = self.cfg.training
        step  = start_step
        accum = 0
        optimizer.zero_grad()
        t0    = time.time()

        for epoch in range(100):   # will break on max_steps
            for batch in loader:
                if step >= max_steps:
                    return step

                ids  = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                lbl  = batch["labels"].to(self.device)

                # Forward pass
                if scaler:
                    try:
                        autocast_context = torch.amp.autocast("cuda")
                    except (AttributeError, TypeError):
                        autocast_context = torch.cuda.amp.autocast()
                    with autocast_context:
                        out  = self.model(ids, mask, lbl)
                        loss = out["loss"] / tc.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    out  = self.model(ids, mask, lbl)
                    loss = out["loss"] / tc.gradient_accumulation_steps
                    loss.backward()

                accum += 1
                if accum % tc.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.max_grad_norm)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1

                    # ── Logging ───────────────────────────────────────────────
                    if step % 50 == 0:
                        elapsed = time.time() - t0
                        lm_l    = out.get("lm_loss", out["loss"]).item()
                        lce_l   = out.get("lce_loss", torch.tensor(0)).item()
                        lr      = scheduler.get_last_lr()[0]
                        print(f"  [{phase}] step={step} | lm={lm_l:.4f} | "
                              f"lce={lce_l:.4f} | lr={lr:.2e} | {elapsed:.0f}s")
                        self.history.append({
                            "step": step, "lm_loss": lm_l, "lce_loss": lce_l, "lr": lr
                        })

                    # ── Eval ──────────────────────────────────────────────────
                    if val_dataset and step % tc.eval_every == 0:
                        val_loss = self._eval(val_dataset)
                        print(f"  📊 Validation loss: {val_loss:.4f}")
                        self.history[-1]["val_loss"] = val_loss

                    # ── Checkpoint ────────────────────────────────────────────
                    if step % tc.save_every == 0:
                        self._save_checkpoint(step)

        return step

    @torch.no_grad()
    def _eval(self, val_dataset) -> float:
        self.model.eval()
        loader = DataLoader(val_dataset, batch_size=self.cfg.training.batch_size)
        total_loss, n = 0.0, 0
        for batch in loader:
            ids  = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            lbl  = batch["labels"].to(self.device)
            out  = self.model(ids, mask, lbl)
            total_loss += out["lm_loss"].item()
            n          += 1
            if n >= 50:
                break
        self.model.train()
        return total_loss / max(n, 1)

    def _save_checkpoint(self, step: int):
        path = self.cfg.paths.checkpoints / f"checkpoint_{step}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        with open(path / "history.json", "w") as f:
            json.dump(self.history, f)
        print(f"  💾 Checkpoint saved: {path}")
