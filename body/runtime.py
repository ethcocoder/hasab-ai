"""
body/runtime.py — Mobile Runtime Inference Engine
═══════════════════════════════════════════════════════════════════
This is what runs ON THE PHONE.

Dependencies at runtime (total ~17MB):
  - onnxruntime        (15MB)
  - numpy              (usually pre-installed)
  - tokenizer_mobile.json (500KB)
  - hasab_amharic.onnx (model file)

NO PyTorch. NO Transformers. NO TensorFlow.
═══════════════════════════════════════════════════════════════════
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict


class MobileRuntime:
    """
    Lightweight inference engine for on-device Amharic generation.
    Uses ONNX Runtime — the only heavy dependency (15MB).
    Everything else is pure Python + numpy.
    """

    def __init__(self, bundle_dir: Path):
        self.bundle_dir = Path(bundle_dir)
        self._session  = None
        self._vocab    = None
        self._id2token = None
        self._special  = None
        self._tokenizer = None
        self._loaded   = False

        # Import ONNX Runtime (only dep)
        try:
            import onnxruntime as ort
            self._ort = ort
        except ImportError:
            raise ImportError(
                "onnxruntime not found.\n"
                "Install with: pip install onnxruntime\n"
                "On mobile: use onnxruntime-mobile (15MB)"
            )

    # ── Load model ────────────────────────────────────────────────────────────

    def load(self):
        model_path = self.bundle_dir / "hasab_amharic.onnx"
        tok_path   = self.bundle_dir / "tokenizer_config.json"
        tokenizer_json = self.bundle_dir / "tokenizer.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # ONNX session — CPU optimized for mobile
        opts = self._ort.SessionOptions()
        opts.intra_op_num_threads = 2       # Limit threads for mobile battery
        opts.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = self._ort.InferenceSession(
            str(model_path),
            sess_options   = opts,
            providers      = ["CPUExecutionProvider"],
        )

        # Load tokenizer vocab
        if tok_path.exists():
            with open(tok_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self._vocab    = config["vocab"]
            self._special  = config["special_tokens"]
            self._id2token = {v: k for k, v in self._vocab.items()}
        # The model is trained with BPE. Prefer the serialized tokenizer so
        # inference uses exactly the same token IDs as training.
        if tokenizer_json.exists():
            try:
                from tokenizers import Tokenizer
                self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
            except ImportError:
                print("⚠️  tokenizers is unavailable; using character fallback")

        self._loaded = True
        print(f"✅ Mobile runtime loaded from {self.bundle_dir}")
        return self

    # ── Tokenize ──────────────────────────────────────────────────────────────

    def encode(self, text: str, max_length: int = 512) -> List[int]:
        if self._tokenizer is not None:
            bos = self._special.get("<bos>", 2)
            return ([bos] + self._tokenizer.encode(text).ids)[:max_length]
        bos = self._special.get("<bos>", 2)
        unk = self._special.get("<unk>", 1)
        pad = self._special.get("<pad>", 0)

        ids = [bos] + [self._vocab.get(ch, unk) for ch in text]
        ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(ids, skip_special_tokens=skip_special)
        special_ids = set(self._special.values()) if skip_special else set()
        tokens = [self._id2token.get(i, "?") for i in ids if i not in special_ids]
        return "".join(tokens)

    # ── Generate ──────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 128,
        temperature:    float = 0.8,
        top_k:          int   = 40,
    ) -> str:
        """
        Autoregressive generation on-device.
        Pure numpy — no PyTorch required.
        """
        if not self._loaded:
            raise RuntimeError("Call .load() first")

        eos_id = self._special.get("<eos>", 3)
        pad_id = self._special.get("<pad>", 0)

        ids = self.encode(prompt)

        for _ in range(max_new_tokens):
            # Prepare inputs as numpy arrays
            inp_ids  = np.array([ids], dtype=np.int64)
            attn_msk = np.ones_like(inp_ids)

            # Run ONNX model
            outputs = self._session.run(
                None,
                {"input_ids": inp_ids, "attention_mask": attn_msk},
            )
            logits = outputs[0][0, -1, :]      # (vocab_size,)

            # Temperature scaling
            logits = logits / max(temperature, 1e-8)

            # Top-k sampling
            if top_k > 0:
                top_k_actual = min(top_k, len(logits))
                top_indices  = np.argpartition(logits, -top_k_actual)[-top_k_actual:]
                mask         = np.full_like(logits, -np.inf)
                mask[top_indices] = logits[top_indices]
                logits = mask

            # Softmax
            logits -= logits.max()
            probs   = np.exp(logits)
            probs  /= probs.sum()

            # Sample
            next_id = int(np.random.choice(len(probs), p=probs))
            ids.append(next_id)

            if next_id == eos_id:
                break

        # Decode (skip input prompt tokens)
        return self.decode(ids[len(self.encode(prompt)):])

    # ── Chat interface ────────────────────────────────────────────────────────

    def chat(self, user_message: str, history: Optional[List[Dict]] = None) -> str:
        """
        Simple chat interface for mobile app integration.
        history: [{"role": "user"|"assistant", "content": str}]
        """
        from soul.safety_gate import SafetyGate
        from heart.tone import ToneModulator
        from mind.cerebellum import Cerebellum

        gate       = SafetyGate()
        tone_mod   = ToneModulator()
        cerebellum = Cerebellum()

        # 1. Safety check
        is_safe, category, refusal = gate.check_input(user_message)
        if not is_safe:
            return refusal

        # 2. Cerebellum fast path
        cached = cerebellum.query(user_message)
        if cached:
            return cached

        # 3. Build prompt
        history = history or []
        prompt_parts = []
        for turn in history[-6:]:   # Last 3 exchanges
            prefix = "<user>" if turn["role"] == "user" else "<assistant>"
            prompt_parts.append(f"{prefix}{turn['content']}")
        prompt_parts.append(f"<user>{user_message}<assistant>")
        full_prompt = "\n".join(prompt_parts)

        # 4. Tone-adjusted decoding params
        tone   = tone_mod.detect_tone(user_message)
        params = tone_mod.get_decoding_params(tone)

        # 5. Generate
        response = self.generate(
            full_prompt,
            max_new_tokens = 128,
            temperature    = params["temperature"],
            top_k          = params["top_k"],
        )

        # 6. Output safety check
        is_safe, sanitized = gate.check_output(response)
        response = sanitized if is_safe else "ይቅርታ፣ ይህን ጥያቄ መመለስ አልቻልኩም።"

        # 7. Learn pattern for cerebellum
        cerebellum.learn(user_message, response)

        return response

    # ── Stats ─────────────────────────────────────────────────────────────────

    def runtime_info(self) -> Dict:
        if not self._loaded:
            return {"loaded": False}
        return {
            "loaded":           True,
            "vocab_size":       len(self._vocab) if self._vocab else 0,
            "providers":        self._session.get_providers(),
            "model_inputs":     [i.name for i in self._session.get_inputs()],
        }
