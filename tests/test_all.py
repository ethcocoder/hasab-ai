"""
tests/test_all.py — Unit tests for all four modules
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch


# ════════════════════════════════════════════════════════════
# SOUL TESTS
# ════════════════════════════════════════════════════════════

class TestSoul:
    def test_core_values(self):
        from soul.values import CoreValues
        cv = CoreValues()
        assert "ሃሳብ" in cv.name
        assert len(cv.ethical_principles) > 0
        assert len(cv.system_prompt) > 100

    def test_safety_gate_blocks_hate(self):
        from soul.safety_gate import SafetyGate
        gate = SafetyGate()
        is_safe, cat, _ = gate.check_input("hate all people")
        assert not is_safe
        assert cat == "hate_speech"

    def test_safety_gate_allows_normal(self):
        from soul.safety_gate import SafetyGate
        gate = SafetyGate()
        is_safe, _, _ = gate.check_input("ሰላም፣ እንዴት ነህ?")
        assert is_safe

    def test_culture_injector(self):
        from soul.culture_injector import CultureInjector
        injector = CultureInjector()
        texts    = ["ሰላም ዓለም"] * 10
        augmented = injector.inject_into_dataset(texts)
        assert len(augmented) > len(texts)
        assert injector.get_random_proverb() != ""

    def test_culture_chat_examples(self):
        from soul.culture_injector import CultureInjector
        injector = CultureInjector(upweight_factor=1)
        examples = injector.build_chat_examples()
        assert len(examples) > 0
        assert "messages" in examples[0]


# ════════════════════════════════════════════════════════════
# HEART TESTS
# ════════════════════════════════════════════════════════════

class TestHeart:
    def test_tone_detection_formal(self):
        from heart.tone import ToneModulator
        tm   = ToneModulator()
        tone = tm.detect_tone("አቶ ተስፋዬ")
        assert tone == "formal"

    def test_tone_detection_empathy(self):
        from heart.tone import ToneModulator
        tm   = ToneModulator()
        tone = tm.detect_tone("I am sad and scared")
        assert tone == "empathetic"

    def test_tone_decoding_params(self):
        from heart.tone import ToneModulator
        tm     = ToneModulator()
        params = tm.get_decoding_params("formal")
        assert "temperature" in params
        assert params["temperature"] < 0.8     # formal = lower temp

    def test_sentiment_fast_score(self):
        from heart.sentiment import SentimentEncoder
        enc   = SentimentEncoder()
        label, conf = enc.fast_score("ደስ አለኝ ጥሩ ጥሩ")
        assert label in ["positive", "very_positive"]
        assert 0 < conf <= 1.0

    def test_context_window(self):
        from heart.context_window import ContextWindowManager
        cwm = ContextWindowManager(max_tokens=128)
        cwm.set_system("You are Hasab.")
        cwm.add_turn("user", "ሰላም")
        cwm.add_turn("assistant", "ሰላም! እንዴት ነህ?")
        msgs = cwm.get_messages()
        assert len(msgs) == 3   # system + 2 turns
        assert msgs[0]["role"] == "system"

    def test_persona_language_detection(self):
        from heart.persona import PersonaTracker
        pt = PersonaTracker()
        pt.update("ሰላም ዓለም ኢትዮጵያ")
        assert pt.state.language == "am"


# ════════════════════════════════════════════════════════════
# MIND TESTS
# ════════════════════════════════════════════════════════════

class TestMind:
    def test_lce_shape(self):
        from mind.lce import LatentCompressionEncoder
        lce    = LatentCompressionEncoder(d_model=768, d_latent=128)
        x      = torch.randn(2, 16, 768)
        out, latent, loss = lce(x, return_latent=True, compute_loss=True)
        assert out.shape    == (2, 16, 768)
        assert latent.shape == (2, 16, 128)
        assert loss.item()  >= 0

    def test_lce_compression_ratio(self):
        from mind.lce import LatentCompressionEncoder
        lce = LatentCompressionEncoder(d_model=768, d_latent=128)
        assert lce.compression_ratio() == 6.0

    def test_lce_encode_only(self):
        from mind.lce import LatentCompressionEncoder
        lce    = LatentCompressionEncoder(d_model=768, d_latent=128)
        x      = torch.randn(1, 8, 768)
        latent = lce.encode_only(x)
        assert latent.shape == (1, 8, 128)

    def test_cerebellum_reflex(self):
        from mind.cerebellum import Cerebellum
        cb     = Cerebellum()
        result = cb.query("ሰላም")
        assert result is not None
        assert len(result) > 0

    def test_cerebellum_learn(self):
        from mind.cerebellum import Cerebellum
        cb = Cerebellum(min_hits_to_cache=2)
        cb.learn("test prompt amharic", "test response", confidence=0.9)
        cb.learn("test prompt amharic", "test response", confidence=0.9)
        result = cb.query("test prompt amharic")
        assert result == "test response"

    def test_cerebellum_stats(self):
        from mind.cerebellum import Cerebellum
        cb    = Cerebellum()
        stats = cb.stats()
        assert "cache_size" in stats
        assert stats["reflex_count"] > 0


# ════════════════════════════════════════════════════════════
# BODY TESTS
# ════════════════════════════════════════════════════════════

class TestBody:
    def test_tokenizer_amharic_detection(self):
        from body.tokenizer import AmharicTokenizer
        assert AmharicTokenizer.is_amharic_text("ሰላም ዓለም")
        assert not AmharicTokenizer.is_amharic_text("Hello World")

    def test_tokenizer_char_fallback(self):
        from body.tokenizer import AmharicTokenizer
        tok = AmharicTokenizer(vocab_size=500)
        tok._train_char_fallback(["ሰላም ዓለም", "አዲስ አበባ", "ኢትዮጵያ"])
        assert tok._trained
        ids = tok.encode("ሰ", add_bos=True, add_eos=True)
        assert ids[0]  == tok.bos_token_id
        assert ids[-1] == tok.eos_token_id

    def test_tokenizer_encode_decode(self):
        from body.tokenizer import AmharicTokenizer
        tok = AmharicTokenizer(vocab_size=500)
        tok._train_char_fallback(["ሰላም ዓለም ኢትዮጵያ"])
        ids     = tok.encode("ሰ", add_bos=False, add_eos=False)
        decoded = tok.decode(ids)
        assert "ሰ" in decoded


# ════════════════════════════════════════════════════════════
# PIPELINE TESTS
# ════════════════════════════════════════════════════════════

class TestPipeline:
    def test_cleaner_normalizes(self):
        from pipeline.clean import AmharicCleaner
        cleaner = AmharicCleaner()
        text    = "  ሰላም    ዓለም  "
        cleaned = cleaner.clean_text(text)
        assert cleaned == "ሰላም ዓለም"

    def test_cleaner_rejects_short(self):
        from pipeline.clean import AmharicCleaner
        cleaner  = AmharicCleaner(min_length=20)
        is_valid, reason = cleaner.is_valid("ሰ")
        assert not is_valid
        assert reason == "too_short"

    def test_cleaner_corpus(self):
        from pipeline.clean import AmharicCleaner
        cleaner = AmharicCleaner(min_length=5, min_amharic_ratio=0.1)
        texts   = ["ሰላም ዓለም", "Hello world", "ኢትዮጵያ ምስራቅ አፍሪካ"]
        clean, stats = cleaner.clean_corpus(texts)
        assert stats["total"] == 3
        assert stats["kept"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
