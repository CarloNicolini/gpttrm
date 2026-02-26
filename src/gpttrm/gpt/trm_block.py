"""
Tiny Recursive Model (TRM) reasoning block adapted for use with continuous
hidden states from a GPT-2 backbone.

This is a simplified, self-contained version of the TRM recursion loop from
the paper "Less is More: Recursive Reasoning with Tiny Networks" (Martineaux 2025).
It operates on continuous hidden state tensors (B, T, D) instead of discrete
token inputs, making it a drop-in "pondering" module between GPT-2 layers.

Key differences from original TRM code:
- No discrete embedding / lm_head (those live in the outer GPT-2 wrapper).
- No puzzle embeddings or sparse embeddings.
- Positional information is already baked into the hidden states from GPT-2.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import RMSNorm from our custom_gpt2 to avoid duplication
from gpttrm.gpt.custom_gpt2 import RMSNorm


@dataclass
class TRMBlockConfig:
    """Configuration for the TRM reasoning block."""

    hidden_size: int = 256  # Must match GPT-2's n_embd
    n_head: int = 4  # Heads for the TRM's internal attention
    expansion: float = 2.0  # MLP expansion factor
    n_reasoning_layers: int = (
        2  # Number of layers per single recursion step (the "tiny" part)
    )
    L_cycles: int = 6  # Inner recursion steps per full cycle
    H_cycles: int = 3  # Outer supervision cycles (T in paper)
    dropout: float = 0.1

    # -- Modernization Toggle --
    modern: bool = False


class TRMAttention(nn.Module):
    """Bidirectional (non-causal) multi-head attention for the TRM block.

    TRM uses non-causal attention because the recursion is refining a *complete*
    hidden state, not generating tokens autoregressively.
    """

    def __init__(self, config: TRMBlockConfig):
        super().__init__()
        assert config.hidden_size % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.hidden_size // config.n_head

        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal attention (is_causal=True) is CRITICAL here to prevent target leakage.
        # If False, the LM will look at future tokens during training loss computation,
        # resulting in artificial ~0 loss and broken autoregressive generation.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.o_proj(y))


class TRMMLP(nn.Module):
    """MLP used inside the TRM reasoning layers. Supports SwiGLU (modern) or GELU (baseline)."""

    def __init__(self, config: TRMBlockConfig):
        super().__init__()
        self.modern = config.modern

        if self.modern:
            # SwiGLU
            inter_size = int(config.expansion * config.hidden_size * 2 / 3)
            inter_size = ((inter_size + 63) // 64) * 64
            self.gate_up_proj = nn.Linear(
                config.hidden_size, inter_size * 2, bias=False
            )
            self.down_proj = nn.Linear(inter_size, config.hidden_size, bias=False)
        else:
            # Standard GPT-2 style MLP
            inter_size = int(config.expansion * config.hidden_size)
            self.c_fc = nn.Linear(config.hidden_size, inter_size)
            self.gelu = nn.GELU(approximate="tanh")
            self.c_proj = nn.Linear(inter_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.modern:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            x = self.down_proj(F.silu(gate) * up)
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)

        return self.dropout(x)


class TRMReasoningLayer(nn.Module):
    """A single transformer-style layer inside the TRM reasoning module.
    GPT-2 Style: Pre-Norm (standard for stability) or TRM Style: Post-Norm (original paper).
    Actually, to match the user's "previous version", I'll keep the Post-Norm logic
    but toggle LayerNorm vs RMSNorm.
    """

    def __init__(self, config: TRMBlockConfig):
        super().__init__()
        self.attn = TRMAttention(config)
        self.mlp = TRMMLP(config)

        if config.modern:
            self.norm1 = RMSNorm(config.hidden_size)
            self.norm2 = RMSNorm(config.hidden_size)
        else:
            self.norm1 = nn.LayerNorm(config.hidden_size)
            self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard Pre-norm: normalize conditioned state, then apply function, add to original residual
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TRMReasoningModule(nn.Module):
    """Sequence of layers applied during one reasoning step."""

    def __init__(self, config: TRMBlockConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TRMReasoningLayer(config) for _ in range(config.n_reasoning_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        for layer in self.layers:
            x = layer(x)
        return x


class TRMBlock(nn.Module):
    """Complete TRM pondering block operating on continuous hidden states."""

    def __init__(self, config: TRMBlockConfig):
        super().__init__()
        self.config = config
        self.reasoning = TRMReasoningModule(config)

        # We no longer use static initialization parameters;
        # y and z are seeded natively in the dual-residual Universal Transformer.

        # ACT Halting Head
        self.halting_head = nn.Linear(config.hidden_size, 1)

        self.proj_in = nn.Identity()
        self.proj_out = nn.Identity()

    def forward(
        self, input_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, T, D = input_embeddings.shape
        x = self.proj_in(input_embeddings)

        # --- Dual-Residual Formulation (Universal Transformer) ---
        # y is the active residual "solution" stream, bridged from GPT-2.
        # z is the latent "memory" stream, initialized to zero.
        y = x.clone()
        z = torch.zeros_like(x)

        if not hasattr(self, "halting_head"):
            self.halting_head = nn.Linear(self.config.hidden_size, 1, device=x.device)

        intermediates = []
        halting_logits = []

        # We unroll exactly L_cycles times, simulating 12 layers (6 cycles * 2 internal layers)
        N = self.config.L_cycles

        for n in range(N):
            # Truncated BPTT: Detach latent states between cycles to prevent
            # unrolled gradient explosions and OOM errors during Deep Supervision.
            if n > 0:
                y = y.detach()
                z = z.detach()

            # 1. Latent recursion
            in_z = x + y + z
            out_z = self.reasoning(in_z)
            z = z + (out_z - in_z)

            # 2. Solution refinement
            in_y = y + z
            out_y = self.reasoning(in_y)
            y = y + (out_y - in_y)

            intermediates.append(y)
            halting_logits.append(self.halting_head(y))

        output = intermediates[-1]

        with torch.no_grad():
            x_norm = x.norm(dim=-1).mean()
            y_norm = y.norm(dim=-1).mean()
            z_norm = z.norm(dim=-1).mean()
            y_delta_norm = (y - x).norm(dim=-1).mean() / (x_norm + 1e-8)
            z_delta_norm = z.norm(dim=-1).mean() / (x_norm + 1e-8)

            y_flat = y.reshape(-1, D)
            z_flat = z.reshape(-1, D)
            cosine_sim = F.cosine_similarity(y_flat, z_flat, dim=-1).mean()

        metrics = {
            "trm/y_norm": y_norm,
            "trm/z_norm": z_norm,
            "trm/y_delta_norm": y_delta_norm,
            "trm/z_delta_norm": z_delta_norm,
            "trm/input_norm": x_norm,
            "trm/y_z_cosine_sim": cosine_sim,
            "trm/intermediates": intermediates,
            "trm/halting_logits": halting_logits,
        }

        return output, metrics
