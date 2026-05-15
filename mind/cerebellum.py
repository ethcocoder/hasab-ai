"""
mind/cerebellum.py — Fast Reflex Cache (Unconscious Automation)
═══════════════════════════════════════════════════════════════════
The cerebellum handles AUTOMATIC responses — patterns so common
that the full transformer doesn't need to run.

Like how humans don't think consciously about walking,
the cerebellum handles: greetings, simple facts, cached answers.

MOBILE BENEFIT:
  Common queries (ሰላም, ምን ዓይነት ነህ?) skip the 150MB model entirely.
  Response time: <5ms vs ~300ms for full inference.
═══════════════════════════════════════════════════════════════════
"""

import hashlib
import time
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class CacheEntry:
    response:   str
    hits:       int   = 0
    created_at: float = field(default_factory=time.time)
    last_hit:   float = field(default_factory=time.time)
    confidence: float = 1.0


# ── Built-in reflex responses (always cached) ─────────────────────────────────
REFLEX_PATTERNS: Dict[str, str] = {
    # Greetings
    "ሰላም": "ሰላም! እንዴት ልረዳህ/ሽ?",
    "hi": "ሰላም! How can I help you?",
    "hello": "ሰላም! How can I help you?",
    "hey": "ሰላም! ምን ልርዳህ/ሽ?",

    # Farewells
    "ደህና ሁን": "ደህና ሁን! ሌላ ጊዜ ደግሞ ትምጣ/ይምጣ።",
    "bye": "Goodbye! Come back anytime.",
    "goodbye": "Goodbye! ደህና ሁን።",

    # Wellbeing
    "እንዴት ነህ": "ጥሩ ነኝ፣ አመሰግናለሁ! አንተስ?",
    "እንዴት ነሽ": "ጥሩ ነኝ፣ አመሰግናለሁ! አንቺስ?",
    "how are you": "I'm doing well, thank you! How about you?",

    # Identity
    "ማን ነህ": "እኔ ሃሳብ ነኝ — የኢትዮጵያ AI ረዳት።",
    "ስምህ ማን ነው": "ስሜ ሃሳብ ነው።",
    "what is your name": "My name is Hasab (ሃሳብ) — an Amharic AI assistant.",
    "who are you": "I am Hasab, an Amharic-first AI assistant built for Ethiopia.",

    # Thanks
    "አመሰግናለሁ": "እንኳን ደስ ያለህ/ሽ! ሌላ ጥያቄ አለህ/ሽ?",
    "thank you": "You're welcome! Anything else I can help with?",
    "thanks": "You're welcome!",
}


class Cerebellum:
    """
    LRU cache of learned + built-in response patterns.
    Mobile-safe: pure Python, no ML at query time.
    """

    def __init__(self, max_size: int = 2048, min_hits_to_cache: int = 3):
        self.max_size         = max_size
        self.min_hits         = min_hits_to_cache
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hit_counts: Dict[str, int] = {}

        # Load built-in reflexes
        self._load_reflexes()

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, text: str) -> Optional[str]:
        """
        Check if a response exists in the reflex cache.
        Returns response string or None if cache miss.
        """
        key = self._key(text)

        # Exact match
        if key in self._cache:
            entry = self._cache[key]
            entry.hits      += 1
            entry.last_hit   = time.time()
            self._cache.move_to_end(key)    # LRU: move to end = most recently used
            return entry.response

        # Fuzzy: check if normalized text starts with a known reflex
        norm = self._normalize(text)
        for pattern, entry in self._cache.items():
            if norm.startswith(pattern[:min(len(norm), 10)]):
                entry.hits += 1
                return entry.response

        return None

    def learn(self, prompt: str, response: str, confidence: float = 0.9):
        """
        Cache a new prompt→response pair after it's been generated.
        Only caches after it's seen min_hits times (avoids caching one-offs).
        """
        key = self._key(prompt)
        self._hit_counts[key] = self._hit_counts.get(key, 0) + 1

        if self._hit_counts[key] >= self.min_hits:
            self._add(key, response, confidence)

    def force_cache(self, prompt: str, response: str):
        """Immediately cache a prompt→response (for built-ins)."""
        self._add(self._key(prompt), response, confidence=1.0)

    def stats(self) -> Dict:
        total_hits = sum(e.hits for e in self._cache.values())
        return {
            "cache_size":    len(self._cache),
            "total_hits":    total_hits,
            "max_size":      self.max_size,
            "reflex_count":  len(REFLEX_PATTERNS),
        }

    def top_patterns(self, n: int = 10) -> List[Tuple[str, int]]:
        return sorted(
            [(k, e.hits) for k, e in self._cache.items()],
            key=lambda x: x[1], reverse=True
        )[:n]

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_reflexes(self):
        for prompt, response in REFLEX_PATTERNS.items():
            self._add(self._key(prompt), response, confidence=1.0)

    def _add(self, key: str, response: str, confidence: float):
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)     # evict LRU
        self._cache[key] = CacheEntry(response=response, confidence=confidence)

    def _key(self, text: str) -> str:
        return self._normalize(text)

    def _normalize(self, text: str) -> str:
        import unicodedata
        text = unicodedata.normalize("NFC", text.strip().lower())
        # Remove punctuation
        text = "".join(c for c in text if c.isalnum() or '\u1200' <= c <= '\u137F' or c == ' ')
        return text.strip()
