"""
heart/context_window.py — Conversation context manager.
Compresses history to stay within token budget on mobile.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Turn:
    role:    str   # "user" | "assistant" | "system"
    content: str
    tokens:  int = 0
    sentiment: str = "neutral"


class ContextWindowManager:
    """
    Manages the conversation window within a fixed token budget.
    Strategy: keep system prompt + last N turns that fit.
    Compresses old turns to summaries when budget exceeded.
    Mobile-safe: pure Python.
    """

    def __init__(self, max_tokens: int = 512, reserve_for_output: int = 256):
        self.max_tokens        = max_tokens
        self.reserve           = reserve_for_output
        self.budget            = max_tokens - reserve_for_output
        self.turns: List[Turn] = []
        self.system_prompt: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def set_system(self, prompt: str):
        self.system_prompt = prompt

    def add_turn(self, role: str, content: str, sentiment: str = "neutral"):
        approx_tokens = len(content.split()) + 4
        self.turns.append(Turn(role=role, content=content,
                               tokens=approx_tokens, sentiment=sentiment))
        self._trim()

    def get_prompt(self, tokenizer=None) -> str:
        """Returns the full prompt string ready for the model."""
        parts = []
        if self.system_prompt:
            parts.append(f"<system>{self.system_prompt}</system>")
        for t in self.turns:
            tag = "<user>" if t.role == "user" else "<assistant>"
            parts.append(f"{tag}{t.content}</{tag[1:]}>")
        return "\n".join(parts)

    def get_messages(self) -> List[Dict]:
        """Returns OpenAI-style messages list."""
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        for t in self.turns:
            msgs.append({"role": t.role, "content": t.content})
        return msgs

    def clear(self):
        self.turns.clear()

    def token_count(self) -> int:
        sys_tokens = len(self.system_prompt.split()) if self.system_prompt else 0
        return sys_tokens + sum(t.tokens for t in self.turns)

    # ── Private ───────────────────────────────────────────────────────────────

    def _trim(self):
        """Remove oldest non-system turns until within budget."""
        while self.token_count() > self.budget and len(self.turns) > 1:
            self.turns.pop(0)


"""
heart/persona.py — User persona state tracker.
Remembers user preferences, name, language, and interaction style.
Mobile-safe: pure Python dict, persists to JSON.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict, field


@dataclass
class PersonaState:
    name:           Optional[str]  = None
    language:       str            = "am"         # am | en | mixed
    preferred_tone: str            = "casual"
    formality:      str            = "auto"       # auto | formal | casual
    topics_liked:   list           = field(default_factory=list)
    turn_count:     int            = 0
    last_sentiment: str            = "neutral"

    # AGI progression: these grow richer over versions
    inferred_age_group: str        = "unknown"    # child | youth | adult | elder
    inferred_expertise: str        = "general"    # general | technical | academic


class PersonaTracker:
    """
    Tracks and evolves the user persona across a conversation.
    v1: Simple heuristic updates.
    v2+ (roadmap): Learn from interaction history with a tiny classifier.
    """

    def __init__(self, save_path: Optional[Path] = None):
        self.state     = PersonaState()
        self.save_path = save_path

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, user_text: str, sentiment: str = "neutral"):
        t = user_text.lower()
        self.state.turn_count   += 1
        self.state.last_sentiment = sentiment

        # Detect name introduction
        for pattern in ["ስሜ", "my name is", "i am", "i'm"]:
            if pattern in t:
                words = user_text.split()
                for i, w in enumerate(words):
                    if pattern in w.lower() and i + 1 < len(words):
                        self.state.name = words[i + 1].strip(".,!?")
                        break

        # Detect language preference
        am_chars = sum(1 for c in user_text if '\u1200' <= c <= '\u137F')
        en_chars  = sum(1 for c in user_text if c.isascii() and c.isalpha())
        if am_chars > en_chars:
            self.state.language = "am"
        elif en_chars > am_chars * 2:
            self.state.language = "en"
        else:
            self.state.language = "mixed"

        # Detect formality from honorifics
        if any(h in user_text for h in ["አቶ", "ወ/ሮ", "ዶ/ር", "ክቡር"]):
            self.state.formality = "formal"

        # Infer age group heuristically
        if any(w in t for w in ["homework", "ቤት ሥራ", "school", "ትምህርት ቤት"]):
            self.state.inferred_age_group = "youth"
        elif any(w in t for w in ["office", "ቢሮ", "business", "ንግድ", "meeting"]):
            self.state.inferred_age_group = "adult"

        if self.save_path:
            self.save(self.save_path)

    def get_tone_recommendation(self) -> str:
        if self.state.formality == "formal":
            return "formal"
        if self.state.last_sentiment in ["negative", "very_negative"]:
            return "empathetic"
        if self.state.inferred_age_group == "youth":
            return "casual"
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
