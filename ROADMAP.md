# 🇪🇹 Hasab (ሃሳብ) — AGI Progression Roadmap

## Vision
Build the first Ethiopian AGI — starting with a mobile Amharic chatbot,
growing into a fully general intelligence that thinks, reasons, and learns
in Amharic-first.

---

## Version Roadmap

### ✅ v1.0 — Chatbot (Current)
**Goal:** Natural Amharic conversation on mobile

| Module  | Capability |
|---------|------------|
| Soul    | Values, cultural DNA, safety gate |
| Heart   | Tone modulation, sentiment, context window |
| Mind    | Qwen2.5-0.5B + LoRA fine-tuning |
| Body    | Custom 8k BPE tokenizer, ONNX export, mobile runtime |

**Metrics target:**
- Perplexity < 50 on Amharic test set
- Model size < 80MB (quantized)
- Response time < 500ms on mid-range Android

---

### 🔜 v2.0 — Reasoning Mind
**Goal:** Multi-step reasoning, chain-of-thought, longer context

New capabilities:
- **Cerebrum v2:** Chain-of-thought reasoning in Amharic
- **Mind:** Extend context to 2048 tokens
- **Heart:** Long-term memory (store/retrieve user facts)
- **Body:** Streaming token output for faster perceived response
- **Soul:** Reinforcement learning from human feedback (RLHF)

---

### 🔮 v3.0 — Multi-Modal Body
**Goal:** Voice + text + image understanding

New capabilities:
- **Body:** Amharic text-to-speech (TTS) integration
- **Body:** Amharic speech-to-text (STT) integration
- **Mind:** Vision encoder for image understanding
- **Soul:** Multimodal cultural grounding
- **Heart:** Voice tone and emotion detection

---

### 🔮 v4.0 — Agentic Mind
**Goal:** Autonomous task completion with tools

New capabilities:
- **Cerebrum v4:** Tool use (web search, calculator, calendar)
- **Mind:** Self-critique and response revision loop
- **Body:** API integration for Ethiopian services
- **Soul:** Expanded constitutional AI rules for agentic behavior

---

### 🔮 v∞ — AGI
**Goal:** General intelligence in Amharic-first

New capabilities:
- **Cerebrum v∞:** Self-improvement loop
- **Mind:** Continual learning from interactions
- **Soul:** Dynamic value updating with human oversight
- **Heart:** Deep empathy and cultural wisdom
- **Body:** Runs on any device, any modality

---

## Architecture Decision Log

### Why LCE (Latent Compression Encoder)?
**Problem:** Transformer hidden states can be too large for mobile RAM.
**Solution:** Bottleneck autoencoder: 768 → 128 → 768.
**Result:** 6x compression of KV cache, ~70% RAM reduction.
**Innovation:** Gated residual mixing — starts as pass-through, learns compression.

### Why custom tokenizer (8k vocab)?
**Problem:** The base tokenizer may be inefficient for Ge'ez → collect Amharic data and measure token efficiency.
**Solution:** BPE trained on Amharic corpus, 8000 vocab.
**Result:** 3-6x fewer tokens for same text → smaller context → faster inference.

### Why ONNX over PyTorch Mobile?
**Problem:** PyTorch Mobile = 30MB+ overhead.
**Solution:** ONNX Runtime Mobile = 15MB, runs anywhere.
**Result:** Smaller app size, faster startup, cross-platform (Android + iOS).

### Why four modules (Soul/Heart/Mind/Body)?
**Design philosophy:** Mirrors human cognition architecture.
- Soul = identity and values (never changes)
- Heart = emotional intelligence (adapts to user)
- Mind = cognitive processing (grows with versions)
- Body = physical form (optimizes for each platform)

This separation makes each module independently upgradeable
without breaking the others — critical for the AGI roadmap.

---

## Contributing

This is an open architecture. Each module can be improved independently:
- **Soul:** Add more cultural training data, refine constitution
- **Heart:** Train better Amharic sentiment classifier
- **Mind:** Improve Qwen LoRA data, evaluation, and mobile quantization
- **Body:** Export to new runtimes (Core ML, TensorRT)

**ዕውቀት አገልግሎት ነው — Knowledge is service.**
