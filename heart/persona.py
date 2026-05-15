"""
heart/persona.py — User persona state tracker.
Remembers user preferences, name, language, and interaction style.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field


@dataclass
class PersonaState:
    name:               Optional[str] = None
    language:           str           = "am"
    preferred_tone:     str           = "casual"
    formality:          str           = "auto"
    topics_liked:       list          = field(default_factory=list)
    turn_count:         int           = 0
    last_sentiment:     str           = "neutral"
    inferred_age_group: str           = "unknown"
    inferred_expertise: str           = "general"


class PersonaTracker:
    def __init__(self, save_path: Optional[Path] = None):
        self.state     = PersonaState()
        self.save_path = save_path

    def update(self, user_text: str, sentiment: str = "neutral"):
        t = user_text.lower()
        self.state.turn_count    += 1
        self.state.last_sentiment = sentiment

        for pattern in ["ስሜ", "my name is", "i am", "i'm"]:
            if pattern in t:
                words = user_text.split()
                for i, w in enumerate(words):
                    if pattern in w.lower() and i + 1 < len(words):
                        self.state.name = words[i + 1].strip(".,!?")
                        break

        am_chars = sum(1 for c in user_text if '\u1200' <= c <= '\u137F')
        en_chars  = sum(1 for c in user_text if c.isascii() and c.isalpha())
        if am_chars > en_chars:           self.state.language = "am"
        elif en_chars > am_chars * 2:     self.state.language = "en"
        else:                             self.state.language = "mixed"

        if any(h in user_text for h in ["አቶ", "ወ/ሮ", "ዶ/ር", "ክቡር"]):
            self.state.formality = "formal"

        if any(w in t for w in ["homework", "ቤት ሥራ", "school"]):
            self.state.inferred_age_group = "youth"
        elif any(w in t for w in ["office", "ቢሮ", "business"]):
            self.state.inferred_age_group = "adult"

        if self.save_path:
            self.save(self.save_path)

    def get_tone_recommendation(self) -> str:
        if self.state.formality == "formal":           return "formal"
        if self.state.last_sentiment in ["negative", "very_negative"]: return "empathetic"
        if self.state.inferred_age_group == "youth":   return "casual"
        return self.state.preferred_tone

    def greet(self) -> str:
        name_part = f" {self.state.name}" if self.state.name else ""
        if self.state.language == "en":
            return f"Hello{name_part}! How can I help you today?"
        return f"ሰላም{name_part}! ዛሬ እንዴት ልረዳህ/ሽ?"

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, ensure_ascii=False, indent=2)

    def load(self, path: Path):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(self.state, k):
                    setattr(self.state, k, v)
