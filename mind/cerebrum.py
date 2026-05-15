"""
mind/cerebrum.py — Cerebrum: Goal Planning & Task Decomposition
═══════════════════════════════════════════════════════════════════
The CONSCIOUS high-level planner. v1 = simple intent classifier.
v2+ roadmap: multi-step reasoning, tool use, self-improvement.

In the mind architecture:
  Cerebellum = fast unconscious reflexes (pattern cache)
  Cerebrum   = slow conscious deliberate planning (this file)

AGI Progression:
  v1: Intent detection + single-turn response (chatbot)
  v2: Multi-turn task planning (chain-of-thought)
  v3: Tool use (search, calculator, calendar)
  v4: Self-critique and response revision
  v∞: Recursive self-improvement
═══════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class Intent(Enum):
    """v1 intent taxonomy for Amharic chatbot."""
    GREETING        = "greeting"
    FAREWELL        = "farewell"
    QUESTION_FACT   = "question_factual"
    QUESTION_OPINION= "question_opinion"
    QUESTION_HOW    = "question_procedural"
    STORYTELLING    = "storytelling"
    TRANSLATION     = "translation"
    MATH            = "math"
    CREATIVE        = "creative_writing"
    COMPLAINT       = "complaint"
    THANKS          = "thanks"
    UNKNOWN         = "unknown"


# Intent detection keyword signals
INTENT_SIGNALS: Dict[Intent, List[str]] = {
    Intent.GREETING:         ["ሰላም", "hello", "hi", "hey", "good morning", "እንደምን"],
    Intent.FAREWELL:         ["ደህና ሁን", "bye", "goodbye", "later", "ቻው"],
    Intent.QUESTION_FACT:    ["ምንድን", "ማን", "መቼ", "የት", "ስንት", "what is", "who is", "when", "where", "how many"],
    Intent.QUESTION_HOW:     ["እንዴት", "how to", "how do", "አሰራር", "ዘዴ"],
    Intent.TRANSLATION:      ["ተርጉም", "translate", "in amharic", "in english", "ትርጉም"],
    Intent.MATH:             ["+", "-", "×", "÷", "=", "calculate", "ስሌት", "ምን ያህል"],
    Intent.CREATIVE:         ["ግጥም", "poem", "story", "ታሪክ", "ዘፈን", "song", "write me"],
    Intent.THANKS:           ["አመሰገን", "thank", "ምስጋና", "አሪፍ", "great"],
    Intent.COMPLAINT:        ["ችግር", "problem", "error", "አልሰራ", "wrong", "doesn't work"],
}


@dataclass
class GoalState:
    """Represents the current goal and plan for a conversation turn."""
    intent:         Intent       = Intent.UNKNOWN
    confidence:     float        = 0.0
    entities:       List[str]    = field(default_factory=list)
    sub_goals:      List[str]    = field(default_factory=list)   # v2+
    requires_tools: List[str]    = field(default_factory=list)   # v3+
    reasoning_steps: List[str]   = field(default_factory=list)   # v2+ CoT


class Cerebrum:
    """
    High-level goal planner and intent classifier.
    v1: Keyword-based intent detection.
    v2+: Will use a small classification head on top of the model.
    Mobile-safe: pure Python in v1.
    """

    def __init__(self, version: int = 1):
        self.version      = version
        self.goal_history: List[GoalState] = []

    # ── Intent Detection ──────────────────────────────────────────────────────

    def detect_intent(self, text: str) -> GoalState:
        """
        v1: Keyword-based intent detection.
        Returns a GoalState with detected intent and confidence.
        """
        t = text.lower().strip()
        best_intent = Intent.UNKNOWN
        best_score  = 0
        best_conf   = 0.0

        for intent, signals in INTENT_SIGNALS.items():
            score = sum(1 for s in signals if s.lower() in t)
            if score > best_score:
                best_score  = score
                best_intent = intent
                best_conf   = min(0.5 + score * 0.15, 0.95)

        # Extract simple entities (capitalized words, Amharic proper nouns)
        entities = self._extract_entities(text)

        goal = GoalState(
            intent     = best_intent,
            confidence = best_conf,
            entities   = entities,
        )

        # v1 sub-goal planning (simple rules)
        goal.sub_goals = self._plan_subgoals(goal)

        self.goal_history.append(goal)
        return goal

    def plan_response_strategy(self, goal: GoalState, persona) -> Dict:
        """
        Returns a strategy dict that guides response generation.
        Grows more sophisticated in v2+.
        """
        strategy = {
            "intent":        goal.intent.value,
            "max_tokens":    self._token_budget(goal.intent),
            "use_cot":       goal.intent in [Intent.MATH, Intent.QUESTION_HOW],  # chain-of-thought
            "tone_hint":     self._tone_for_intent(goal.intent, persona),
            "entities":      goal.entities,
        }
        return strategy

    # ── v2+ Stubs (roadmap) ───────────────────────────────────────────────────

    def chain_of_thought(self, problem: str) -> List[str]:
        """
        v2: Multi-step reasoning decomposition.
        Currently returns a simple 3-step template.
        v2 will use the model itself to generate reasoning steps.
        """
        return [
            f"1. Understand the question: {problem[:50]}...",
            "2. Recall relevant knowledge",
            "3. Formulate answer in Amharic",
        ]

    def self_critique(self, response: str) -> Optional[str]:
        """
        v4: Model critiques its own output and revises.
        Stub for now — returns None (no revision).
        """
        return None

    # ── Private ───────────────────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction: capitalized words + known Amharic nouns."""
        words    = text.split()
        entities = [w for w in words if w and (w[0].isupper() or '\u1200' <= w[0] <= '\u137F')]
        return list(set(entities))[:5]

    def _plan_subgoals(self, goal: GoalState) -> List[str]:
        mapping = {
            Intent.TRANSLATION:    ["detect_source_language", "translate", "format_output"],
            Intent.QUESTION_HOW:   ["understand_topic", "recall_steps", "explain_clearly"],
            Intent.MATH:           ["parse_expression", "compute", "explain_answer"],
            Intent.CREATIVE:       ["understand_style", "generate_draft", "refine"],
        }
        return mapping.get(goal.intent, ["understand", "respond"])

    def _token_budget(self, intent: Intent) -> int:
        budgets = {
            Intent.GREETING:     32,
            Intent.FAREWELL:     16,
            Intent.THANKS:       24,
            Intent.QUESTION_FACT: 128,
            Intent.QUESTION_HOW: 256,
            Intent.MATH:         128,
            Intent.CREATIVE:     384,
            Intent.TRANSLATION:  128,
        }
        return budgets.get(intent, 128)

    def _tone_for_intent(self, intent: Intent, persona) -> str:
        if intent == Intent.GREETING:   return "casual"
        if intent == Intent.CREATIVE:   return "poetic"
        if intent == Intent.COMPLAINT:  return "empathetic"
        return persona.get_tone_recommendation() if persona else "casual"
