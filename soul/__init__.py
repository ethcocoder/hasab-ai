"""
soul/__init__.py — Identity & Purpose Layer
The soul is baked into training data and system prompts.
Zero extra parameters at runtime.
"""

from .values import CoreValues
from .culture_injector import CultureInjector
from .safety_gate import SafetyGate

__all__ = ["CoreValues", "CultureInjector", "SafetyGate"]
