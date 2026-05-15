"""
pipeline/clean.py — Amharic text cleaning & normalization
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Tuple


class AmharicCleaner:
    """
    Cleans raw Amharic text for training.
    Handles Ge'ez script normalization, deduplication, filtering.
    """

    GEEZ_RANGE = (0x1200, 0x137F)

    # Common Amharic punctuation normalizations
    PUNCT_MAP = {
        "፡": "፡",    "።": "።",    "፣": "፣",
        "，": "፣",    "。": "።",    "！": "!",    "？": "?",
    }

    def __init__(
        self,
        min_length:      int   = 20,
        max_length:      int   = 512,
        min_amharic_ratio: float = 0.30,
        dedup:           bool  = True,
    ):
        self.min_length         = min_length
        self.max_length         = max_length
        self.min_amharic_ratio  = min_amharic_ratio
        self.dedup              = dedup
        self._seen: set         = set()

    def clean_text(self, text: str) -> str:
        # Unicode normalization
        text = unicodedata.normalize("NFC", text)
        # Normalize punctuation
        for src, dst in self.PUNCT_MAP.items():
            text = text.replace(src, dst)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove lines that are mostly numbers/symbols
        text = re.sub(r"^[\d\s\W]+$", "", text, flags=re.MULTILINE)
        return text

    def is_valid(self, text: str) -> Tuple[bool, str]:
        """Returns (is_valid, reason)"""
        if len(text) < self.min_length:
            return False, "too_short"
        if len(text) > self.max_length:
            text = text[:self.max_length]   # truncate rather than reject
        am_ratio = sum(1 for c in text if self._is_amharic(c)) / max(len(text), 1)
        if am_ratio < self.min_amharic_ratio:
            return False, "low_amharic_ratio"
        return True, "ok"

    def clean_corpus(self, texts: List[str]) -> Tuple[List[str], dict]:
        stats = {"total": len(texts), "kept": 0, "removed_short": 0,
                 "removed_ratio": 0, "removed_dup": 0}
        result = []
        for text in texts:
            text = self.clean_text(text)
            valid, reason = self.is_valid(text)
            if not valid:
                stats[f"removed_{reason}"] = stats.get(f"removed_{reason}", 0) + 1
                continue
            if self.dedup:
                key = text[:100]
                if key in self._seen:
                    stats["removed_dup"] += 1
                    continue
                self._seen.add(key)
            result.append(text)
            stats["kept"] += 1
        return result, stats

    def split_into_sentences(self, text: str) -> List[str]:
        """Split Amharic text at sentence boundaries."""
        sentences = re.split(r"[።\n]+", text)
        return [s.strip() for s in sentences if len(s.strip()) >= self.min_length]

    def _is_amharic(self, ch: str) -> bool:
        cp = ord(ch)
        return self.GEEZ_RANGE[0] <= cp <= self.GEEZ_RANGE[1]

    def process_file(self, input_path: Path, output_path: Path) -> dict:
        with open(input_path, "r", encoding="utf-8") as f:
            raw = f.read()
        paragraphs = [p for p in raw.split("\n\n") if p.strip()]
        sentences  = []
        for p in paragraphs:
            sentences.extend(self.split_into_sentences(p))
        cleaned, stats = self.clean_corpus(sentences)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned))
        print(f"✅ Cleaned: {stats['kept']}/{stats['total']} sentences → {output_path}")
        return stats
