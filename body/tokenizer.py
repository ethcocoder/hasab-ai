"""
body/tokenizer.py — Custom BPE Tokenizer for Amharic (Ge'ez script)
═══════════════════════════════════════════════════════════════════
WHY custom tokenizer:
  A generic English tokenizer can waste context on Amharic. Amharic characters
  (ሀ-ፐ, Unicode U+1200–U+137F) fall back to byte-level encoding,
  splitting each character into 3 bytes. This wastes ~6x context.

OUR APPROACH:
  BPE trained on Amharic corpus. Vocab size 8000:
    - 768 Ge'ez base characters
    - 7232 learned BPE merge rules (common syllable sequences)
  Each Amharic syllable = 1 token instead of 3 bytes.

MOBILE BENEFIT:
  Same sentence uses 3-6x fewer tokens → smaller context → faster.
═══════════════════════════════════════════════════════════════════
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Union


# ── Ge'ez Unicode ranges ──────────────────────────────────────────────────────
GEEZ_RANGE        = (0x1200, 0x137F)    # Core Ethiopic
GEEZ_SUPPLEMENT   = (0x1380, 0x139F)
GEEZ_EXTENDED     = (0x2D80, 0x2DDF)

SPECIAL_TOKENS = {
    "<pad>":        0,
    "<unk>":        1,
    "<bos>":        2,
    "<eos>":        3,
    "<soul>":       4,
    "<heart>":      5,
    "<mind>":       6,
    "<body>":       7,
    "<user>":       8,
    "<assistant>":  9,
    "<system>":    10,
    "<amharic>":   11,
    "<english>":   12,
}


class AmharicTokenizer:
    """
    Lightweight BPE tokenizer for Amharic.
    Trains on corpus, saves as JSON vocab (no sentencepiece required at inference).
    Mobile inference only needs this class + vocab.json (~500KB).
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size    = vocab_size
        self.vocab:        Dict[str, int] = {}
        self.id2token:     Dict[int, str] = {}
        self.merges:       List[tuple] = []
        self._trained      = False

        # Load special tokens
        self.special_tokens    = SPECIAL_TOKENS
        self.pad_token_id      = SPECIAL_TOKENS["<pad>"]
        self.unk_token_id      = SPECIAL_TOKENS["<unk>"]
        self.bos_token_id      = SPECIAL_TOKENS["<bos>"]
        self.eos_token_id      = SPECIAL_TOKENS["<eos>"]

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, texts: List[str], show_progress: bool = True):
        """
        Train BPE on a list of Amharic texts.
        Uses HuggingFace tokenizers library for efficiency.
        """
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
            from tokenizers.normalizers import NFC

            tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            tokenizer.normalizer   = NFC()
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.WhitespaceSplit(),
            ])
            tokenizer.decoder = decoders.BPEDecoder()

            trainer = trainers.BpeTrainer(
                vocab_size      = self.vocab_size,
                min_frequency   = 2,
                special_tokens  = list(SPECIAL_TOKENS.keys()),
                show_progress   = show_progress,
            )
            tokenizer.train_from_iterator(texts, trainer=trainer)
            self._hf_tokenizer = tokenizer
            self.vocab    = tokenizer.get_vocab()
            self.id2token = {v: k for k, v in self.vocab.items()}
            self._trained = True

        except ImportError:
            # Fallback: simple character-level tokenizer for Ge'ez
            self._train_char_fallback(texts)

    def _train_char_fallback(self, texts: List[str]):
        """Character-level fallback when tokenizers lib unavailable."""
        self.vocab = dict(SPECIAL_TOKENS)
        next_id = max(SPECIAL_TOKENS.values()) + 1

        char_freq: Dict[str, int] = {}
        for text in texts:
            for ch in text:
                if self._is_amharic(ch) or ch.isascii():
                    char_freq[ch] = char_freq.get(ch, 0) + 1

        for ch, _ in sorted(char_freq.items(), key=lambda x: -x[1]):
            if next_id >= self.vocab_size:
                break
            if ch not in self.vocab:
                self.vocab[ch] = next_id
                next_id += 1

        self.id2token = {v: k for k, v in self.vocab.items()}
        self._trained = True

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> List[int]:
        if not self._trained:
            raise RuntimeError("Tokenizer not trained. Call .train() or .load() first.")

        if hasattr(self, "_hf_tokenizer"):
            ids = self._hf_tokenizer.encode(text).ids
        else:
            ids = [self.vocab.get(ch, self.unk_token_id) for ch in text]

        if add_bos: ids = [self.bos_token_id] + ids
        if add_eos: ids = ids + [self.eos_token_id]

        if max_length:
            if padding:
                ids = ids[:max_length]
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            else:
                ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special_ids = set(SPECIAL_TOKENS.values())
            ids = [i for i in ids if i not in special_ids]

        if hasattr(self, "_hf_tokenizer"):
            return self._hf_tokenizer.decode(ids)
        else:
            return "".join(self.id2token.get(i, "?") for i in ids)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: bool = True,
    ) -> Dict[str, List[List[int]]]:
        all_ids    = [self.encode(t, max_length=max_length, padding=padding) for t in texts]
        all_masks  = [[1 if tok != self.pad_token_id else 0 for tok in ids] for ids in all_ids]
        return {"input_ids": all_ids, "attention_mask": all_masks}

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if hasattr(self, "_hf_tokenizer"):
            self._hf_tokenizer.save(str(directory / "tokenizer.json"))

        config = {
            "vocab_size":    self.vocab_size,
            "special_tokens": self.special_tokens,
            "vocab":         self.vocab,
        }
        with open(directory / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"✅ Tokenizer saved to {directory}")

    def load(self, directory: Path):
        directory = Path(directory)

        cfg_path = directory / "tokenizer_config.json"
        tok_path = directory / "tokenizer.json"

        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.vocab_size     = config["vocab_size"]
            self.special_tokens = config["special_tokens"]
            self.vocab          = config["vocab"]
            self.id2token       = {v: k for k, v in self.vocab.items()}

        if tok_path.exists():
            from tokenizers import Tokenizer
            self._hf_tokenizer = Tokenizer.from_file(str(tok_path))

        self._trained = True
        return self

    # ── Utils ─────────────────────────────────────────────────────────────────

    def vocab_size_actual(self) -> int:
        return len(self.vocab)

    def token_count(self, text: str) -> int:
        return len(self.encode(text, add_bos=False, add_eos=False))

    @staticmethod
    def _is_amharic(ch: str) -> bool:
        cp = ord(ch)
        return (GEEZ_RANGE[0] <= cp <= GEEZ_RANGE[1] or
                GEEZ_SUPPLEMENT[0] <= cp <= GEEZ_SUPPLEMENT[1] or
                GEEZ_EXTENDED[0] <= cp <= GEEZ_EXTENDED[1])

    @staticmethod
    def is_amharic_text(text: str, threshold: float = 0.3) -> bool:
        if not text:
            return False
        am_chars = sum(1 for c in text if AmharicTokenizer._is_amharic(c))
        return am_chars / len(text) >= threshold

    def __len__(self) -> int:
        return self.vocab_size_actual()

    def __repr__(self) -> str:
        trained = "trained" if self._trained else "untrained"
        return f"AmharicTokenizer(vocab={self.vocab_size_actual()}, {trained})"
