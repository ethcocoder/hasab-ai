"""
pipeline/finetune.py — Qwen2.5-0.5B LoRA fine-tuning.

This is the only language-model training backend in Hasab. It uses the
pretrained Qwen tokenizer and applies LoRA adapters so a Colab T4 can run a
small smoke test without updating all 0.5B base-model weights.
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model


class QwenTextDataset(Dataset):
    """Tokenized causal-language-model examples with masked padding labels."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.examples = []
        for text in texts:
            encoded = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class QwenTrainer:
    """LoRA trainer for Qwen2.5-0.5B."""

    def __init__(self, config, device: Optional[str] = None):
        self.cfg = config
        configured = getattr(config.training, "device", "auto")
        self.device = device or (
            "cuda" if configured == "auto" and torch.cuda.is_available()
            else "cpu" if configured == "auto"
            else configured
        )
        self.model_name = config.model.base_model
        self.output_dir = Path(config.paths.qwen_adapter)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_model = None
        self._loaded_tokenizer = None
        print(f"🔧 Qwen training on: {self.device}")
        print(f"   Base model: {self.model_name}")

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self):
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )
        model.config.pad_token_id = model.config.eos_token_id
        model.config.use_cache = False
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.cfg.model.lora_r,
            lora_alpha=self.cfg.model.lora_alpha,
            lora_dropout=self.cfg.model.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        if self.device == "cuda":
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        return model.to(self.device)

    def train(self, train_texts: List[str], val_texts: Optional[List[str]] = None):
        tokenizer = self._load_tokenizer()
        train_ds = QwenTextDataset(train_texts, tokenizer, self.cfg.model.max_seq_len)
        val_ds = QwenTextDataset(val_texts or [], tokenizer, self.cfg.model.max_seq_len)
        if not train_ds:
            raise ValueError("No training examples were created from the corpus")

        model = self._load_model()
        tc = self.cfg.training
        loader = DataLoader(
            train_ds,
            batch_size=tc.batch_size,
            shuffle=True,
            num_workers=tc.num_workers,
            pin_memory=self.device == "cuda",
        )
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda") if self.device == "cuda" else None
        max_steps = tc.max_steps
        accum_steps = tc.gradient_accumulation_steps
        step = 0
        accum = 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.time()

        print(f"   Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")
        print(f"   Max steps: {max_steps} | Effective batch: {tc.batch_size * accum_steps}")

        while step < max_steps:
            for batch in loader:
                if step >= max_steps:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if scaler:
                    with torch.amp.autocast("cuda"):
                        loss = model(**batch).loss / accum_steps
                    scaler.scale(loss).backward()
                else:
                    loss = model(**batch).loss / accum_steps
                    loss.backward()
                accum += 1
                if accum % accum_steps != 0:
                    continue

                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step == 1 or step % 10 == 0:
                    elapsed = time.time() - started
                    print(f"   step={step}/{max_steps} loss={loss.item() * accum_steps:.4f} elapsed={elapsed:.0f}s")

        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        with open(self.output_dir / "hasab_qwen_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "base_model": self.model_name,
                "max_seq_len": self.cfg.model.max_seq_len,
                "train_examples": len(train_ds),
                "max_steps": max_steps,
            }, f, indent=2)
        print(f"✅ Qwen LoRA adapter saved to {self.output_dir}")
        return model, tokenizer

    def load_adapter(self):
        if self._loaded_model is not None:
            return self._loaded_model, self._loaded_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        model = PeftModel.from_pretrained(base, self.output_dir).to(self.device)
        model.eval()
        self._loaded_model = model
        self._loaded_tokenizer = tokenizer
        return model, tokenizer

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128):
        model, tokenizer = self.load_adapter()
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
