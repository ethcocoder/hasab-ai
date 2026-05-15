"""
heart/tone.py — Tone modulation for Amharic responses.
Adjusts decoding temperature, top-k, and system prompt prefix
based on detected conversation register.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ToneProfile:
    name:           str
    temperature:    float
    top_k:          int
    top_p:          float
    rep_penalty:    float
    prompt_prefix:  str


TONE_PROFILES: Dict[str, ToneProfile] = {
    "formal": ToneProfile(
        name="formal",
        temperature=0.6,
        top_k=30,
        top_p=0.85,
        rep_penalty=1.3,
        prompt_prefix="[ፎርማል] ",
    ),
    "casual": ToneProfile(
        name="casual",
        temperature=0.9,
        top_k=60,
        top_p=0.95,
        rep_penalty=1.1,
        prompt_prefix="[ቀላል] ",
    ),
    "empathetic": ToneProfile(
        name="empathetic",
        temperature=0.75,
        top_k=40,
        top_p=0.90,
        rep_penalty=1.2,
        prompt_prefix="[ርኅሩህ] ",
    ),
    "informative": ToneProfile(
        name="informative",
        temperature=0.5,
        top_k=20,
        top_p=0.80,
        rep_penalty=1.4,
        prompt_prefix="[መረጃ] ",
    ),
    "poetic": ToneProfile(
        name="poetic",
        temperature=1.1,
        top_k=80,
        top_p=0.98,
        rep_penalty=1.0,
        prompt_prefix="[ግጥም] ",
    ),
}

# ── Keyword signals for auto-detection ───────────────────────────────────────
FORMAL_SIGNALS    = ["አቶ", "ወ/ሮ", "ዶ/ር", "ፕሮፌሰር", "ክቡር", "Dr.", "Mr.", "Mrs.", "please", "kindly"]
EMPATHY_SIGNALS   = ["ሐዘን", "ጭንቀት", "ፍርሀት", "ታመምኩ", "ረዳ", "sad", "help", "scared", "sick", "cry"]
QUESTION_SIGNALS  = ["ምንድን", "እንዴት", "ለምን", "መቼ", "ማን", "what", "how", "why", "when", "who"]
CREATIVE_SIGNALS  = ["ግጥም", "ታሪክ", "ዘፈን", "poem", "story", "song", "creative", "write me"]


class ToneModulator:
    """
    Detects the appropriate tone from user input and conversation history,
    and returns matching decoding parameters.
    Mobile-safe: pure Python, no ML.
    """

    def __init__(self, default_tone: str = "casual"):
        self.default_tone  = default_tone
        self.profiles      = TONE_PROFILES
        self._current_tone = default_tone

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_tone(self, user_text: str, history_turns: int = 0) -> str:
        """
        Auto-detects tone from user message.
        Returns one of: formal | casual | empathetic | informative | poetic
        """
        t = user_text.lower()

        if any(s.lower() in t for s in EMPATHY_SIGNALS):
            tone = "empathetic"
        elif any(s.lower() in t for s in FORMAL_SIGNALS):
            tone = "formal"
        elif any(s.lower() in t for s in CREATIVE_SIGNALS):
            tone = "poetic"
        elif any(s.lower() in t for s in QUESTION_SIGNALS):
            tone = "informative"
        else:
            tone = self.default_tone

        self._current_tone = tone
        return tone

    def get_profile(self, tone: Optional[str] = None) -> ToneProfile:
        tone = tone or self._current_tone
        return self.profiles.get(tone, self.profiles[self.default_tone])

    def get_decoding_params(self, tone: Optional[str] = None) -> Dict:
        """Returns kwargs ready to pass to model.generate()"""
        p = self.get_profile(tone)
        return {
            "temperature":          p.temperature,
            "top_k":                p.top_k,
            "top_p":                p.top_p,
            "repetition_penalty":   p.rep_penalty,
        }

    def current_tone(self) -> str:
        return self._current_tone
