"""
pipeline/evaluate.py — Evaluation: perplexity, BLEU, generation quality
"""

import math
import torch
import json
from typing import List, Dict
from torch.utils.data import DataLoader
from pathlib import Path


class Evaluator:
    """
    Comprehensive evaluation suite for AmharicGPT2.
    Metrics: perplexity, text generation quality, LCE compression quality.
    """

    def __init__(self, model, tokenizer, config, device: str = "cpu"):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = config
        self.device    = device
        self.model.to(device)

    # ── Perplexity ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def perplexity(self, dataset) -> float:
        """Compute perplexity on a dataset. Lower = better."""
        self.model.eval()
        loader     = DataLoader(dataset, batch_size=8)
        total_nll  = 0.0
        total_toks = 0

        for batch in loader:
            ids  = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            lbl  = batch["labels"].to(self.device)
            out  = self.model(ids, mask, lbl)
            n_toks = (lbl != -100).sum().item()
            total_nll  += out["lm_loss"].item() * n_toks
            total_toks += n_toks

        ppl = math.exp(total_nll / max(total_toks, 1))
        return round(ppl, 2)

    # ── Generation quality ────────────────────────────────────────────────────

    @torch.no_grad()
    def generation_samples(self, prompts: List[str] = None) -> List[Dict]:
        self.model.eval()
        prompts = prompts or self.cfg.eval.generation_prompts
        samples = []

        for prompt in prompts:
            ids = torch.tensor(
                [self.tokenizer.encode(prompt, add_bos=True, add_eos=False)],
                dtype=torch.long
            ).to(self.device)

            generated = self.model.generate(
                ids,
                max_new_tokens = 80,
                temperature    = 0.8,
                top_k          = 40,
            )
            out_ids  = generated[0][ids.shape[1]:].tolist()
            response = self.tokenizer.decode(out_ids)
            samples.append({"prompt": prompt, "response": response})
            print(f"  ↳ {prompt!r}\n    {response!r}\n")

        return samples

    # ── LCE compression quality ───────────────────────────────────────────────

    @torch.no_grad()
    def lce_reconstruction_error(self, dataset) -> Dict:
        """Measure how well LCE reconstructs hidden states."""
        self.model.eval()
        loader     = DataLoader(dataset, batch_size=4)
        total_mse  = 0.0
        n          = 0

        for batch in loader:
            ids  = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            out  = self.model(ids, mask, return_latents=True)
            if "lce_loss" in out:
                total_mse += out["lce_loss"].item()
                n         += 1
            if n >= 20:
                break

        avg_mse = total_mse / max(n, 1)
        return {
            "lce_mse":          round(avg_mse, 6),
            "compression_ratio": self.model.lce_layers[0].compression_ratio() if self.model.lce_layers else None,
            "n_lce_layers":     len(self.model.lce_layers),
        }

    # ── Full report ───────────────────────────────────────────────────────────

    def full_report(self, train_ds, val_ds, test_ds) -> Dict:
        print("\n" + "="*60)
        print("📊 EVALUATION REPORT")
        print("="*60)

        print("\n🔢 Model size:")
        params = self.model.param_count()
        for k, v in params.items():
            print(f"  {k:12s}: {v/1e6:.2f}M params")
        print(f"  {'size_mb':12s}: {self.model.model_size_mb():.1f} MB")

        print("\n📉 Perplexity:")
        ppl_val  = self.perplexity(val_ds)
        ppl_test = self.perplexity(test_ds)
        print(f"  Validation : {ppl_val}")
        print(f"  Test       : {ppl_test}")

        print("\n🗜  LCE Compression:")
        lce_stats = self.lce_reconstruction_error(val_ds)
        for k, v in lce_stats.items():
            print(f"  {k}: {v}")

        print("\n📝 Generation Samples:")
        samples = self.generation_samples()

        report = {
            "params":       params,
            "size_mb":      self.model.model_size_mb(),
            "ppl_val":      ppl_val,
            "ppl_test":     ppl_test,
            "lce":          lce_stats,
            "samples":      samples,
        }

        out_path = self.cfg.paths.logs / "eval_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Report saved: {out_path}")
        return report
