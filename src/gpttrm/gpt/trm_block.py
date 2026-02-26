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

    def forward(self, x: torch.Tensor, injection: torch.Tensor = None) -> torch.Tensor:
        # We form the conditioned state h for the sub-layers to read from.
        # This prevents the injection (e.g. z_H + x) from accumulating geometrically
        # inside the residual stream over dozens of recurrent cycles.
        h = x if injection is None else x + injection

        # Pre-norm: normalize conditioned state, then apply function, add to original residual
        x = x + self.attn(self.norm1(h))

        h = x if injection is None else x + injection
        x = x + self.mlp(self.norm2(h))
        return x


class TRMReasoningModule(nn.Module):
    """Sequence of layers applied during one reasoning step."""

    def __init__(self, config: TRMBlockConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TRMReasoningLayer(config) for _ in range(config.n_reasoning_layers)]
        )

    def forward(
        self, hidden_states: torch.Tensor, injection: torch.Tensor
    ) -> torch.Tensor:
        x = hidden_states
        for layer in self.layers:
            x = layer(x, injection=injection)
        return x


class TRMBlock(nn.Module):
    """Complete TRM pondering block operating on continuous hidden states."""

    def __init__(self, config: TRMBlockConfig):
        super().__init__()
        self.config = config
        self.reasoning = TRMReasoningModule(config)

        # We no longer use static initialization parameters;
        # z_L and z_H are seeded directly from the input sequence x.
        self.gate = nn.Parameter(torch.tensor(-2.0))

        self.proj_in = nn.Identity()
        self.proj_out = nn.Identity()

    def forward(
        self, input_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, T, D = input_embeddings.shape
        x = self.proj_in(input_embeddings)

        # --- Residual Seeding (Paper Implementation) ---
        # Instead of static noise, we seed the internal reasoning (z_L) and
        # current solution guess (z_H) with the actual contextualized input (x).
        # This bridges GPT-2's latent space seamlessly into the TRM block.
        z_H = x.clone()
        z_L = x.clone()

        with torch.no_grad():
            for _h in range(self.config.H_cycles - 1):
                for _l in range(self.config.L_cycles):
                    z_L = self.reasoning(z_L, z_H + x)
                z_H = self.reasoning(z_H, z_L)

        for _l in range(self.config.L_cycles):
            z_L = self.reasoning(z_L, z_H + x)
        z_H = self.reasoning(z_H, z_L)

        alpha = torch.sigmoid(self.gate)
        output = self.proj_out(alpha * z_H + (1.0 - alpha) * x)

        with torch.no_grad():
            x_norm = x.norm(dim=-1).mean()
            z_H_norm = z_H.norm(dim=-1).mean()
            delta = z_H - x
            z_H_delta_norm = delta.norm(dim=-1).mean() / (x_norm + 1e-8)
            z_L_flat = z_L.reshape(-1, D)
            z_H_flat = z_H.reshape(-1, D)
            cosine_sim = F.cosine_similarity(z_L_flat, z_H_flat, dim=-1).mean()
            output_delta_norm = (output - x).norm(dim=-1).mean() / (x_norm + 1e-8)

        metrics = {
            "trm/gate_alpha": alpha.detach(),
            "trm/gate_raw": self.gate.detach(),
            "trm/z_H_delta_norm": z_H_delta_norm,
            "trm/z_H_norm": z_H_norm,
            "trm/input_norm": x_norm,
            "trm/z_L_z_H_cosine_sim": cosine_sim,
            "trm/gated_output_delta_norm": output_delta_norm,
        }

        return output, metrics
