"""
config.py — Central configuration for Amharic Mobile-AGI
All hyperparameters, paths, and module settings live here.
Change values here; never hardcode in modules.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ─────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()


@dataclass
class PathConfig:
    root:               Path = ROOT
    data_raw:           Path = ROOT / "data" / "raw"
    data_clean:         Path = ROOT / "data" / "clean"
    data_processed:     Path = ROOT / "data" / "processed"
    checkpoints:        Path = ROOT / "models" / "checkpoints"
    qwen_adapter:       Path = ROOT / "models" / "qwen_adapter"
    exported:           Path = ROOT / "models" / "exported"
    logs:               Path = ROOT / "logs"
    tokenizer_dir:      Path = ROOT / "models" / "tokenizer"
    soul_config:        Path = ROOT / "soul" / "values.json"

    def make_dirs(self):
        for f in self.__dataclass_fields__:
            p = getattr(self, f)
            if isinstance(p, Path) and not p.suffix:
                p.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    # Sources for Amharic text acquisition
    wikipedia_lang:         str  = "am"
    oscar_lang:             str  = "am"
    min_sentence_length:    int  = 20
    max_sentence_length:    int  = 512
    dedup_threshold:        float = 0.85   # Jaccard similarity for dedup
    min_amharic_ratio:      float = 0.70   # Min fraction of Ge'ez chars
    max_raw_mb:             int  = 500     # Stop downloading after this
    train_split:            float = 0.90
    val_split:              float = 0.05
    test_split:             float = 0.05


@dataclass
class TokenizerConfig:
    vocab_size:     int  = 8000    # retained for legacy data utilities
    min_frequency:  int  = 2
    special_tokens: List[str] = field(default_factory=lambda: [
        "<pad>", "<unk>", "<bos>", "<eos>",
        "<soul>", "<heart>", "<mind>", "<body>",   # module tokens
        "<user>", "<assistant>", "<system>",
        "<amharic>", "<english>",                  # language tags
    ])
    model_type:     str  = "bpe"


@dataclass
class ModelConfig:
    """Qwen2.5 causal language model and LoRA settings."""
    base_model:         str   = "Qwen/Qwen2.5-0.5B"
    max_seq_len:        int   = 512
    lora_r:             int   = 8
    lora_alpha:         int   = 16
    lora_dropout:       float = 0.05


@dataclass
class HeartConfig:
    """Emotional intelligence layer settings"""
    sentiment_hidden:   int   = 64
    sentiment_classes:  int   = 5              # very_neg/neg/neutral/pos/very_pos
    tone_styles:        List[str] = field(default_factory=lambda: [
        "formal", "casual", "empathetic", "informative", "poetic"
    ])
    context_window:     int   = 2048
    persona_dim:        int   = 32


@dataclass
class SoulConfig:
    """Identity & safety settings"""
    amharic_culture_weight: float = 2.0        # Upweight cultural training data
    safety_categories: List[str] = field(default_factory=lambda: [
        "hate_speech", "violence", "misinformation", "self_harm"
    ])
    constitutional_rules_file: str = "soul/constitution.txt"
    refusal_confidence_threshold: float = 0.85


@dataclass
class TrainingConfig:
    batch_size:         int   = 8
    gradient_accumulation_steps: int = 4      # Effective batch = 32
    learning_rate:      float = 3e-4
    weight_decay:       float = 0.01
    warmup_steps:       int   = 500
    max_steps:          int   = 50_000
    eval_every:         int   = 500
    save_every:         int   = 1000
    max_grad_norm:      float = 1.0
    fp16:               bool = True
    device:             str  = "auto"        # auto | cuda | cpu
    seed:               int   = 42
    num_workers:        int   = 2

@dataclass
class BodyConfig:
    """Output, quantization & mobile export settings"""
    # Decoding
    max_new_tokens:     int   = 256
    temperature:        float = 0.8
    top_k:              int   = 50
    top_p:              float = 0.92
    repetition_penalty: float = 1.2
    pad_token_id:       int   = 0

    # Quantization
    quantize_method:    str   = "dynamic"      # dynamic | static | qat
    quantize_dtype:     str   = "int8"

    # Export
    export_formats:     List[str] = field(default_factory=lambda: ["onnx", "tflite"])
    onnx_opset:         int   = 14
    mobile_max_ram_mb:  int   = 150            # Hard limit for mobile

    # Runtime (no heavy libs)
    runtime_packages:   List[str] = field(default_factory=lambda: [
        "onnxruntime",          # 15MB — only dep at inference
        "sentencepiece",        # 1MB
        "numpy",                # already on device usually
    ])


@dataclass
class EvalConfig:
    batch_size:         int   = 16
    max_eval_samples:   int   = 1000
    perplexity_stride:  int   = 512
    bleu_ngrams:        int   = 4
    generation_prompts: List[str] = field(default_factory=lambda: [
        "ሰላም፣ እንዴት ነህ?",
        "ኢትዮጵያ",
        "አማርኛ ቋንቋ",
        "ዛሬ ምን ትፈልጋለህ?",
        "አዲስ አበባ ከተማ",
    ])


# ─────────────────────────────────────────────────────────────
# MASTER CONFIG
# ─────────────────────────────────────────────────────────────
class Config:
    def __init__(self):
        self.paths      = PathConfig()
        self.data       = DataConfig()
        self.tokenizer  = TokenizerConfig()
        self.model      = ModelConfig()
        self.heart      = HeartConfig()
        self.soul       = SoulConfig()
        self.training   = TrainingConfig()
        self.body       = BodyConfig()
        self.eval       = EvalConfig()

    def make_dirs(self):
        self.paths.make_dirs()
        (self.paths.root / "models" / "tokenizer").mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return (
            f"Config(\n"
            f"  base_model={self.model.base_model}, "
            f"max_seq={self.model.max_seq_len}, "
            f"batch={self.training.batch_size}, "
            f"lr={self.training.learning_rate}\n"
            f")"
        )


CFG = Config()
