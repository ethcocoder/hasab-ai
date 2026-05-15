"""
heart/sentiment.py — Lightweight Amharic sentiment tagging.
BiLSTM head attached to GPT-2 output. ~0.3MB extra on device.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


SENTIMENT_LABELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]
AMHARIC_SENTIMENT_LEXICON = {
    # Positive
    "ደስ": 2, "ምስጋና": 2, "አመሰገን": 2, "ፍቅር": 2, "ጥሩ": 1, "ውብ": 2,
    "አስደሳች": 2, "ስኬት": 2, "ተስፋ": 1, "ኩራት": 1,
    # Negative
    "ሐዘን": -2, "ብስጭት": -1, "ፍርሀት": -2, "ቁጣ": -2, "ጥላቻ": -2,
    "ሕመም": -1, "ከፋ": -2, "አስቸጋሪ": -1, "ጭንቀት": -2,
    # English
    "happy": 2, "great": 2, "love": 2, "sad": -2, "hate": -2,
    "angry": -2, "fear": -2, "good": 1, "bad": -1,
}


class SentimentHead(nn.Module):
    """
    Tiny BiLSTM sentiment classifier that sits on top of GPT-2 hidden states.
    Input:  (batch, seq_len, d_model=768)
    Output: (batch, 5) logits
    """

    def __init__(self, d_model: int = 768, hidden: int = 64, num_classes: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout  = nn.Dropout(0.2)
        self.fc       = nn.Linear(hidden * 2, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, d_model)
        out, _ = self.lstm(hidden_states)          # (B, T, 2*hidden)
        pooled  = out.mean(dim=1)                  # (B, 2*hidden)
        return self.fc(self.dropout(pooled))        # (B, num_classes)


class SentimentEncoder:
    """
    Runtime sentiment detector.
    Fast path: lexicon lookup (no GPU, <0.1ms).
    Slow path: SentimentHead (requires model hidden states).
    """

    def __init__(self):
        self.lexicon = AMHARIC_SENTIMENT_LEXICON
        self.labels  = SENTIMENT_LABELS
        self.head: SentimentHead = None   # set after model loads

    def fast_score(self, text: str) -> Tuple[str, float]:
        """
        Lexicon-based fast sentiment. Returns (label, confidence).
        Mobile default path — no model needed.
        """
        words = text.lower().split()
        score = sum(self.lexicon.get(w, 0) for w in words)
        score = max(-4, min(4, score))

        if score <= -3:   return "very_negative", 0.9
        elif score <= -1: return "negative",      0.75
        elif score >= 3:  return "very_positive", 0.9
        elif score >= 1:  return "positive",      0.75
        else:             return "neutral",        0.6

    def model_score(self, hidden_states: torch.Tensor) -> Tuple[str, float]:
        """Model-based sentiment (used during training evaluation)."""
        if self.head is None:
            raise RuntimeError("SentimentHead not attached. Call .set_head() first.")
        logits = self.head(hidden_states)
        probs  = torch.softmax(logits, dim=-1)
        idx    = probs.argmax(dim=-1).item()
        return self.labels[idx], probs[0, idx].item()

    def set_head(self, head: SentimentHead):
        self.head = head
