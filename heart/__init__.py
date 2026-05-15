"""
heart/__init__.py — Emotional & Contextual Intelligence Layer
"""
from .tone import ToneModulator
from .sentiment import SentimentEncoder
from .context_window import ContextWindowManager
from .persona import PersonaTracker

__all__ = ["ToneModulator", "SentimentEncoder", "ContextWindowManager", "PersonaTracker"]
