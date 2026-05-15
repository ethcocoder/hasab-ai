"""
mind/model.py — AmharicGPT2: GPT-2 + Latent Compression Encoder
═══════════════════════════════════════════════════════════════════
Architecture:
  GPT-2 transformer with LCE bottlenecks inserted every N layers.
  LCE compresses hidden states before they flow through attention.

  Layer 0  → Attention → FFN → hidden_0
  LCE_0:    hidden_0 → latent_0 (128d) → hidden_0' (gated)
  Layer 3  → Attention → FFN → hidden_3
  LCE_1:    hidden_3 → latent_1 → hidden_3'
  ...
  LM Head  → logits → token probabilities
═══════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from transformers import GPT2LMHeadModel, GPT2Config
from .lce import LatentCompressionEncoder
from .cerebellum import Cerebellum


class AmharicGPT2(nn.Module):
    """
    GPT-2 base + custom Amharic tokenizer embeddings + LCE bottlenecks.

    Key design choices:
      1. Resize token embeddings to match 8k Amharic vocab
      2. Insert LCE every `lce_insert_every_n` transformer layers
      3. Cerebellum for cached fast responses
      4. Pruned attention heads for mobile efficiency
    """

    def __init__(
        self,
        vocab_size:         int   = 8000,
        d_model:            int   = 768,
        n_layers:           int   = 12,
        n_heads:            int   = 12,
        max_seq_len:        int   = 512,
        lce_enabled:        bool  = True,
        lce_latent_dim:     int   = 128,
        lce_insert_every_n: int   = 3,
        lce_dropout:        float = 0.1,
        cerebellum_enabled: bool  = True,
        pretrained:         str   = "gpt2",         # load pretrained weights
    ):
        super().__init__()

        self.vocab_size         = vocab_size
        self.d_model            = d_model
        self.lce_enabled        = lce_enabled
        self.lce_insert_every_n = lce_insert_every_n

        # ── 1. Load GPT-2 base ────────────────────────────────────────────────
        cfg = GPT2Config(
            vocab_size     = vocab_size,
            n_embd         = d_model,
            n_layer        = n_layers,
            n_head         = n_heads,
            n_positions    = max_seq_len,
            n_ctx          = max_seq_len,
            bos_token_id   = 2,
            eos_token_id   = 3,
        )

        if pretrained:
            self.gpt2 = GPT2LMHeadModel.from_pretrained(pretrained)
            self.gpt2.resize_token_embeddings(vocab_size)
            # Update config to match ours
            self.gpt2.config.n_positions = max_seq_len
            self.gpt2.config.n_ctx       = max_seq_len
        else:
            self.gpt2 = GPT2LMHeadModel(cfg)

        # ── 2. LCE bottlenecks — inserted between transformer layers ──────────
        self.lce_layers: nn.ModuleList = nn.ModuleList()
        if lce_enabled:
            n_lce = n_layers // lce_insert_every_n
            for i in range(n_lce):
                lce = LatentCompressionEncoder(
                    d_model   = d_model,
                    d_latent  = lce_latent_dim,
                    dropout   = lce_dropout,
                    layer_id  = i * lce_insert_every_n,
                )
                self.lce_layers.append(lce)

        # ── 3. Sentiment head (Heart module hook) ─────────────────────────────
        from heart.sentiment import SentimentHead
        self.sentiment_head = SentimentHead(d_model=d_model)

        # ── 4. Cerebellum ─────────────────────────────────────────────────────
        self.cerebellum = Cerebellum() if cerebellum_enabled else None

        # Accumulated LCE reconstruction loss (cleared each forward pass)
        self._lce_loss: Optional[torch.Tensor] = None

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      (B, T) token ids
            attention_mask: (B, T) 1=real, 0=pad
            labels:         (B, T) for language modeling loss
            return_latents: whether to return LCE latent vectors

        Returns dict with keys:
            loss:           combined LM + LCE loss (if labels given)
            lm_loss:        language modeling loss only
            lce_loss:       reconstruction loss only
            logits:         (B, T, vocab_size)
            latents:        list of (B, T, d_latent) tensors (if return_latents)
        """
        latents = []
        lce_loss_total = torch.tensor(0.0, device=input_ids.device)
        lce_idx = 0

        # ── Run through GPT-2 transformer layers with sequential LCE hooks ──
        device = input_ids.device
        
        # 1. Embeddings
        inputs_embeds = self.gpt2.transformer.wte(input_ids)
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=device)
        position_embeds = self.gpt2.transformer.wpe(position_ids.unsqueeze(0))
        hidden_states = self.gpt2.transformer.drop(inputs_embeds + position_embeds)
        
        # 2. Attention mask
        if attention_mask is not None:
            extended_attention_mask = self.gpt2.get_extended_attention_mask(attention_mask, input_ids.size(), device)
        else:
            extended_attention_mask = None

        # 3. Iterate through blocks
        for i, block in enumerate(self.gpt2.transformer.h):
            outputs = block(
                hidden_states,
                attention_mask=extended_attention_mask,
            )
            hidden_states = outputs[0]

            # Apply LCE bottleneck sequentially
            if self.lce_enabled and (i + 1) % self.lce_insert_every_n == 0 and lce_idx < len(self.lce_layers):
                hidden_states, latent, lce_loss = self.lce_layers[lce_idx](
                    hidden_states, return_latent=return_latents, compute_loss=(labels is not None)
                )
                if latent is not None:
                    latents.append(latent)
                if lce_loss is not None:
                    lce_loss_total = lce_loss_total + lce_loss
                lce_idx += 1

        # 4. Final LayerNorm
        hidden_states = self.gpt2.transformer.ln_f(hidden_states)

        # ── LM head: project to vocab ─────────────────────────────────────────
        logits = self.gpt2.lm_head(hidden_states)   # (B, T, vocab_size)

        # ── Compute losses ────────────────────────────────────────────────────
        result = {"logits": logits}
        lm_loss = torch.tensor(0.0)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )
            lce_weight = 0.1
            total_loss = lm_loss + lce_weight * lce_loss_total
            result.update({
                "loss":     total_loss,
                "lm_loss":  lm_loss,
                "lce_loss": lce_loss_total,
            })

        if return_latents:
            result["latents"] = latents

        # ── Sentiment (attach to result for Heart module) ─────────────────────
        result["sentiment_logits"] = self.sentiment_head(hidden_states)

        return result

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids:          torch.Tensor,
        max_new_tokens:     int   = 256,
        temperature:        float = 0.8,
        top_k:              int   = 50,
        top_p:              float = 0.92,
        repetition_penalty: float = 1.2,
        eos_token_id:       int   = 3,
        pad_token_id:       int   = 0,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-k + top-p + repetition penalty.
        Runs in eval mode — no gradients computed.
        """
        self.eval()
        generated = input_ids.clone()
        past_tokens = set(input_ids[0].tolist())

        for _ in range(max_new_tokens):
            out = self.forward(generated)
            next_logits = out["logits"][:, -1, :] / temperature   # (B, vocab)

            # Repetition penalty
            for tok_id in past_tokens:
                if tok_id < next_logits.size(-1):
                    next_logits[:, tok_id] /= repetition_penalty

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, -1:]] = -float("Inf")

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cumulative_probs > top_p] = -float("Inf")
                next_logits.scatter_(1, sorted_idx, sorted_logits)

            probs    = torch.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_tok], dim=1)
            past_tokens.add(next_tok.item())

            if next_tok.item() == eos_token_id:
                break

        return generated

    # ── Utilities ─────────────────────────────────────────────────────────────

    def param_count(self) -> Dict[str, int]:
        gpt2_params = sum(p.numel() for p in self.gpt2.parameters())
        lce_params  = sum(p.numel() for lce in self.lce_layers for p in lce.parameters())
        sent_params = sum(p.numel() for p in self.sentiment_head.parameters())
        return {
            "gpt2":      gpt2_params,
            "lce":       lce_params,
            "sentiment": sent_params,
            "total":     gpt2_params + lce_params + sent_params,
        }

    def model_size_mb(self) -> float:
        total = sum(p.numel() * p.element_size() for p in self.parameters())
        return total / (1024 ** 2)

    def freeze_gpt2(self):
        """Freeze GPT-2 base weights, only train LCE + embeddings."""
        for p in self.gpt2.parameters():
            p.requires_grad = False
        # Unfreeze embeddings so new Amharic tokens are learned
        for p in self.gpt2.transformer.wte.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def prune_attention_heads(self, sparsity: float = 0.3):
        """
        Prune the lowest-magnitude attention heads.
        Reduces computation by ~30% with minimal quality loss.
        """
        heads_to_prune: Dict[int, List[int]] = {}
        for layer_i, block in enumerate(self.gpt2.transformer.h):
            attn   = block.attn
            n_head = attn.num_heads
            # Score heads by weight norm
            w      = attn.c_attn.weight   # (d_model, 3*d_model)
            head_size = self.d_model // n_head
            scores = []
            for h in range(n_head):
                start = h * head_size
                end   = start + head_size
                norm  = w[:, start:end].norm().item()
                scores.append((norm, h))
            scores.sort()
            n_prune = int(n_head * sparsity)
            heads_to_prune[layer_i] = [h for _, h in scores[:n_prune]]

        self.gpt2.prune_heads(heads_to_prune)
        return heads_to_prune
