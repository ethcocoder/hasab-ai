"""
body/quantize.py — Model quantization for mobile deployment.
INT8 dynamic quantization — no heavy libs at runtime.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict


class ModelQuantizer:
    """
    Quantizes a trained causal language model to INT8 for mobile.
    Dynamic quantization: weights stored as INT8, activations quantized on-the-fly.
    No calibration dataset required.
    """

    def __init__(self, method: str = "dynamic"):
        self.method = method

    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply dynamic INT8 quantization."""
        model.eval()

        if self.method == "dynamic":
            quantized = torch.quantization.quantize_dynamic(
                model,
                qconfig_spec={
                    nn.Linear:  torch.quantization.default_dynamic_qconfig,
                    nn.LSTM:    torch.quantization.default_dynamic_qconfig,
                },
                dtype=torch.qint8,
            )
            print("✅ Dynamic INT8 quantization applied")
            return quantized

        raise ValueError(f"Unknown quantization method: {self.method}")

    def size_comparison(self, original: nn.Module, quantized: nn.Module) -> Dict:
        def model_size_mb(m):
            return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6

        orig_size  = model_size_mb(original)
        quant_size = model_size_mb(quantized)
        return {
            "original_mb":  round(orig_size, 1),
            "quantized_mb": round(quant_size, 1),
            "reduction":    f"{(1 - quant_size/orig_size)*100:.1f}%",
        }
