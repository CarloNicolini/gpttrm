import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class CustomGPT2Config:
    vocab_size: int = 50257
    seq_len: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1
    # Use True if you want to tie the word embeddings to the final lm_head weights
    tie_word_embeddings: bool = True

    # -- Modernization Flags --
    modern: bool = False
    n_kv_head: Optional[int] = None  # For GQA (Grouped Query Attention)
    rope_theta: float = 10000.0  # Base frequency for RoPE


# ---------------------------------------------------------------------------
# Modern Components: RMSNorm, RoPE, SwiGLU
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (used in Llama/SmolLM)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryPositionalEmbeddings(nn.Module):
    """Simple RoPE implementation."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q, k):
        # q, k: (B, H, T, D)
        T = q.shape[2]
        cos = self.cos_cached[:, :, :T, :]
        sin = self.sin_cached[:, :, :T, :]

        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rope = (q * cos) + (rotate_half(q) * sin)
        k_rope = (k * cos) + (rotate_half(k) * sin)
        return q_rope, k_rope


# ---------------------------------------------------------------------------
# GPT-2 Components (Updated with Modern support)
# ---------------------------------------------------------------------------


class CustomGPT2Attention(nn.Module):
    def __init__(self, config: CustomGPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # GQA Support
        self.n_kv_head = (
            config.n_kv_head if (config.modern and config.n_kv_head) else config.n_head
        )
        self.n_groups = self.n_head // self.n_kv_head

        # Projections
        self.q_proj = nn.Linear(
            config.n_embd, config.n_head * self.head_dim, bias=not config.modern
        )
        self.k_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=not config.modern
        )
        self.v_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=not config.modern
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=not config.modern)

        # RoPE
        if config.modern:
            self.rope = RotaryPositionalEmbeddings(
                self.head_dim, config.seq_len, config.rope_theta
            )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask (fallback for non-flash)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.seq_len, config.seq_len)).view(
                1, 1, config.seq_len, config.seq_len
            ),
        )

    def forward(self, x):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.config.modern:
            q, k = self.rope(q, k)

            # GQA: Repeat k, v groups if necessary
            if self.n_kv_head != self.n_head:
                k = k.repeat_interleave(self.n_groups, dim=1)
                v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled Dot Product Attention
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True if not self.config.modern else True,  # Always causal for GPT
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class CustomGPT2MLP(nn.Module):
    def __init__(self, config: CustomGPT2Config):
        super().__init__()
        self.modern = config.modern

        if self.modern:
            # SwiGLU: Uses 3 projections (gate, up, down)
            # Standard hidden dim is often 8/3 * n_embd for SwiGLU to match param count of 4 * n_embd GELU
            # But we'll keep it simple: use 2/3 * (4 * n_embd) to keep parity or just stay with 4.
            hidden_dim = 4 * config.n_embd
            self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
            self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
            self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.gelu = nn.GELU(approximate="tanh")
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.modern:
            # SwiGLU(x) = (SiLU(xW) * xV)W_down
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            x = F.silu(gate) * up
            x = self.down_proj(x)
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)

        x = self.dropout(x)
        return x


class CustomGPT2Block(nn.Module):
    def __init__(self, config: CustomGPT2Config):
        super().__init__()
        if config.modern:
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.ln_2 = nn.LayerNorm(config.n_embd)

        self.attn = CustomGPT2Attention(config)
        self.mlp = CustomGPT2MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CustomGPT2Model(nn.Module):
    def __init__(self, config: CustomGPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [CustomGPT2Block(config) for _ in range(config.n_layer)]
                ),
                ln_f=RMSNorm(config.n_embd)
                if config.modern
                else nn.LayerNorm(config.n_embd),
            )
        )

        # Only add absolute PEs if NOT modern (RoPE replaces them)
        if not config.modern:
            self.transformer.wpe = nn.Embedding(config.seq_len, config.n_embd)

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("down_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Normal init for projections
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_layer: int = 0,
        end_layer: Optional[int] = None,
    ) -> torch.Tensor:
        device = idx.device if idx is not None else inputs_embeds.device

        if (idx is not None) and (inputs_embeds is None):
            b, t = idx.size()
            x = self.transformer.wte(idx)

            # Add absolute PEs if NOT modern
            if not self.config.modern:
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                pos_emb = self.transformer.wpe(pos)
                x = x + pos_emb

            x = self.transformer.drop(x)
        elif (inputs_embeds is not None) and (idx is None):
            x = inputs_embeds
            # If start_layer == 0 and NOT modern, we might need PEs
            # But usually we pass hidden states that already have them.
            if start_layer == 0 and not self.config.modern:
                b, t, _ = inputs_embeds.size()
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                pos_emb = self.transformer.wpe(pos)
                x = self.transformer.drop(inputs_embeds + pos_emb)
        else:
            raise ValueError("Pass either idx or inputs_embeds, not both.")

        # run the transformer blocks
        if end_layer is None:
            end_layer = self.config.n_layer

        for block in self.transformer.h[start_layer:end_layer]:
            x = block(x)

        # Only final-norm if we are at the very end
        if end_layer == self.config.n_layer:
            x = self.transformer.ln_f(x)

        return x
