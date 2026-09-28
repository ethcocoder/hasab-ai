"""
body/export.py — Export to ONNX and TFLite for mobile deployment.
No PyTorch or Transformers needed at runtime after export.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import json


class MobileExporter:
    """
    Exports the trained causal language model to:
      1. ONNX      → runs on onnxruntime-mobile (Android/iOS)
      2. TFLite    → runs on TFLite runtime (Android)
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── ONNX Export ───────────────────────────────────────────────────────────

    def export_onnx(
        self,
        model:      nn.Module,
        tokenizer,
        seq_len:    int = 64,
        opset:      int = 14,
    ) -> Path:
        """Export to ONNX with dynamic sequence length."""
        try:
            import onnx
            import onnxruntime
        except ImportError:
            raise ImportError("pip install onnx onnxruntime")

        model.eval()
        out_path = self.output_dir / "hasab_amharic.onnx"

        dummy_ids  = torch.zeros(1, seq_len, dtype=torch.long)
        dummy_mask = torch.ones(1, seq_len, dtype=torch.long)

        # Wrap model for ONNX (forward with just input_ids + attention_mask)
        class ONNXWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, input_ids, attention_mask):
                out = self.m(input_ids=input_ids, attention_mask=attention_mask)
                return out["logits"]

        wrapper = ONNXWrapper(model)

        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            str(out_path),
            opset_version = opset,
            input_names   = ["input_ids", "attention_mask"],
            output_names  = ["logits"],
            dynamic_axes  = {
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits":         {0: "batch", 1: "seq"},
            },
            do_constant_folding = True,
        )

        # Validate
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)

        size_mb = out_path.stat().st_size / 1e6
        print(f"✅ ONNX exported: {out_path} ({size_mb:.1f}MB)")
        return out_path

    # ── TFLite Export (via ONNX → TF → TFLite) ───────────────────────────────

    def export_tflite(self, onnx_path: Path) -> Optional[Path]:
        """Convert ONNX → TFLite (requires onnx-tf)."""
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf
        except ImportError:
            print("⚠️  TFLite export requires: pip install onnx-tf tensorflow")
            print("    Skipping TFLite — ONNX export is sufficient for Android.")
            return None

        out_path = self.output_dir / "hasab_amharic.tflite"
        tf_dir   = self.output_dir / "_tf_intermediate"

        onnx_model = onnx.load(str(onnx_path))
        tf_rep     = prepare(onnx_model)
        tf_rep.export_graph(str(tf_dir))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(out_path, "wb") as f:
            f.write(tflite_model)

        size_mb = out_path.stat().st_size / 1e6
        print(f"✅ TFLite exported: {out_path} ({size_mb:.1f}MB)")
        return out_path

    # ── Save tokenizer for mobile ─────────────────────────────────────────────

    def export_tokenizer(self, tokenizer) -> Path:
        """Save lightweight tokenizer vocab for mobile runtime."""
        tok_path = self.output_dir / "tokenizer_mobile.json"
        tokenizer.save(self.output_dir)
        print(f"✅ Mobile tokenizer saved: {tok_path}")
        return tok_path

    # ── Mobile bundle ─────────────────────────────────────────────────────────

    def create_bundle(self, model_path: Path, tokenizer) -> Path:
        """Create a single deployable bundle with model + tokenizer + config."""
        import zipfile

        self.export_tokenizer(tokenizer)

        meta = {
            "model_name":  "hasab_amharic_v1",
            "version":     "1.0.0",
            "language":    "am",
            "runtime":     "onnxruntime",
            "runtime_pkg": "onnxruntime-mobile",
            "min_runtime_mb": 15,
            "max_ram_mb":  150,
        }
        with open(self.output_dir / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        bundle_path = self.output_dir / "hasab_mobile_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in self.output_dir.glob("*"):
                if f.is_file() and f.suffix != ".zip":
                    zf.write(f, f.name)

        size_mb = bundle_path.stat().st_size / 1e6
        print(f"\n📦 Mobile bundle: {bundle_path} ({size_mb:.1f}MB)")
        print(f"   Deploy this on Android/iOS using onnxruntime-mobile")
        return bundle_path
