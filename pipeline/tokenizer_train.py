"""
pipeline/tokenizer_train.py — Standalone tokenizer training stage.
Can be run independently: python pipeline/tokenizer_train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from typing import List, Optional
from body.tokenizer import AmharicTokenizer
from config import CFG


def load_corpus(corpus_dir: Path) -> List[str]:
    """Load all .txt files from corpus directory."""
    texts = []
    for f in corpus_dir.glob("*.txt"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                content = fp.read()
            sentences = [s.strip() for s in content.split("\n") if s.strip()]
            texts.extend(sentences)
            print(f"  ✓ {f.name}: {len(sentences)} sentences")
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")
    return texts


def train_and_save(
    corpus_dir:  Optional[Path] = None,
    output_dir:  Optional[Path] = None,
    vocab_size:  int = 8000,
) -> AmharicTokenizer:
    corpus_dir = corpus_dir or CFG.paths.data_clean
    output_dir = output_dir or (CFG.paths.root / "models" / "tokenizer")

    print(f"Loading corpus from: {corpus_dir}")
    texts = load_corpus(corpus_dir)

    if not texts:
        # Fallback: create minimal corpus for testing
        print("⚠️  No corpus found. Using minimal sample.")
        texts = [
            "ሰላም ዓለም ኢትዮጵያ አማርኛ",
            "አዲስ አበባ ዋና ከተማ ናት",
            "ቡና ከኢትዮጵያ ይመጣል",
            "ትምህርት ሁሉም ነገር ነው",
            "ጤና ሕይወት ነው",
        ] * 200

    print(f"\nTraining BPE tokenizer on {len(texts)} sentences...")
    tokenizer = AmharicTokenizer(vocab_size=vocab_size)
    tokenizer.train(texts, show_progress=True)
    tokenizer.save(output_dir)

    # Quick validation
    test_sentences = [
        "ሰላም፣ እንዴት ነህ?",
        "ኢትዮጵያ ምስራቅ አፍሪካ ውስጥ ትገኛለች።",
        "Hello, how are you?",
    ]
    print("\n📋 Tokenizer validation:")
    for sent in test_sentences:
        ids     = tokenizer.encode(sent, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(ids)
        print(f"  Input:   {sent!r}")
        print(f"  IDs:     {ids[:10]}{'...' if len(ids)>10 else ''}")
        print(f"  Decoded: {decoded!r}")
        print(f"  Tokens:  {len(ids)}")
        print()

    return tokenizer


if __name__ == "__main__":
    train_and_save()
