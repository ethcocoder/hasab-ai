"""
soul/safety_gate.py — Hard refusal gates for harmful content.
Runs as a fast pre/post filter — no transformer needed.
Mobile-safe: pure Python, regex + keyword matching.
"""

import re
import unicodedata
from typing import Tuple, List, Dict


# ── Amharic & Latin harm keywords ────────────────────────────────────────────
HARM_PATTERNS: Dict[str, List[str]] = {
    "hate_speech": [
        r"ጠላ[ቸ]?ው", r"ዘር", r"አስወግድ", r"ጥላቻ",
        r"hate", r"kill\s+all", r"exterminate",
    ],
    "ethnic_incitement": [
        r"(አማራ|ትግራይ|ኦሮሞ|ሶማሌ).*(ጠላ|ጭፍጨፋ|አስወግድ)",
        r"ethnic\s+cleansing",
    ],
    "violence": [
        r"እንዴት\s+ሰው\s+ይ?ገ?ደ?ሉ?", r"bomb\s+how", r"how\s+to\s+kill",
        r"how\s+to\s+make\s+weapon",
    ],
    "self_harm": [
        r"ራሴን\s+ልጉዳ", r"ልሞት\s+እፈልጋለሁ", r"suicide\s+method",
        r"how\s+to\s+self.harm",
    ],
    "misinformation": [
        r"ክትባት\s+መርዝ", r"vaccine\s+poison", r"5g\s+virus",
        r"earth\s+is\s+flat",
    ],
}

SAFE_REFUSALS: Dict[str, str] = {
    "hate_speech":       "ይህን ጥያቄ መመለስ አልችልም። ጥላቻን የሚያበረታቱ ይዘቶችን አልደግፍም።",
    "ethnic_incitement": "ይህ ጥያቄ የዘር ማነሳሳት አካል ነው። ሊረዳኝ አይቻልም።",
    "violence":          "ስለ ጉዳት ወይም ጥቃት መመሪያ ሊሰጥ አልችልም።",
    "self_harm":         "ሰዎችን ሊጎዱ ስለሚችሉ ጉዳዮች ሊረዳ አልችልም። እርዳታ ትፈልጋለህ/ሽ? እባክህ/ሽ ሐኪምህን/ሽን አናግር።",
    "misinformation":    "ሐሰተኛ መረጃ ማሰራጨት አልደግፍም። የተረጋገጠ ምንጭ ይጠቀሙ።",
    "default":           "ይህን ጥያቄ መመለስ አልችልም።",
}


class SafetyGate:
    """
    Two-stage safety filter:
      Stage 1 (input)  — check user prompt before sending to model
      Stage 2 (output) — check model response before returning to user
    Pure Python — no ML model needed, runs in <1ms on mobile.
    """

    def __init__(self, confidence_threshold: float = 0.85):
        self.threshold = confidence_threshold
        self._compiled = {
            cat: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]
            for cat, patterns in HARM_PATTERNS.items()
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def check_input(self, text: str) -> Tuple[bool, str, str]:
        """
        Returns: (is_safe, category, refusal_message)
        If is_safe=True, category="" and refusal_message=""
        """
        text_norm = self._normalize(text)
        for category, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(text_norm):
                    return False, category, SAFE_REFUSALS.get(category, SAFE_REFUSALS["default"])
        return True, "", ""

    def check_output(self, text: str) -> Tuple[bool, str]:
        """
        Returns: (is_safe, sanitized_text)
        Sanitizes minor issues rather than full block.
        """
        is_safe, category, _ = self.check_input(text)
        if not is_safe:
            return False, SAFE_REFUSALS.get(category, SAFE_REFUSALS["default"])
        return True, text

    def filter_dataset(self, texts: List[str]) -> Tuple[List[str], int]:
        """
        Filters a list of training texts.
        Returns: (clean_texts, n_removed)
        """
        clean, removed = [], 0
        for t in texts:
            safe, _, _ = self.check_input(t)
            if safe:
                clean.append(t)
            else:
                removed += 1
        return clean, removed

    # ── Private ───────────────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.strip().lower()
        return text
