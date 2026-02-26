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
- Uses standard LayerNorm instead of RMSNorm for consistency with GPT-2.
- Positional information is already baked into the hidden states from GPT-2.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


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

        # Non-causal attention (is_causal=False)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.o_proj(y))


class TRMMLP(nn.Module):
    """SwiGLU-style MLP used inside the TRM reasoning layers."""

    def __init__(self, config: TRMBlockConfig):
        super().__init__()
        inter_size = int(config.expansion * config.hidden_size * 2 / 3)
        # Round up to nearest multiple of 64 for efficiency
        inter_size = ((inter_size + 63) // 64) * 64

        self.gate_up_proj = nn.Linear(config.hidden_size, inter_size * 2, bias=False)
        self.down_proj = nn.Linear(inter_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TRMReasoningLayer(nn.Module):
    """A single transformer-style layer inside the TRM reasoning module.
    Uses Post-Norm (as in the original TRM paper) with residual connections.
    """

    def __init__(self, config: TRMBlockConfig):
        super().__init__()
        self.attn = TRMAttention(config)
        self.mlp = TRMMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-norm: add residual then normalize (matching TRM paper)
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x


class TRMReasoningModule(nn.Module):
    """Stack of TRM reasoning layers that form one recursion step.

    Following the paper: this module takes a hidden state and an injection
    signal, sums them, and passes through the reasoning layers.
    """

    def __init__(self, config: TRMBlockConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TRMReasoningLayer(config) for _ in range(config.n_reasoning_layers)]
        )

    def forward(
        self, hidden_states: torch.Tensor, injection: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: The state being refined (z_L or z_H).
            injection: Additive signal (e.g. z_H + input_emb for z_L update,
                       or z_L for z_H update).
        Returns:
            Refined hidden states.
        """
        x = hidden_states + injection
        for layer in self.layers:
            x = layer(x)
        return x


class TRMBlock(nn.Module):
    """Complete TRM pondering block operating on continuous hidden states.

    Implements the full recursion loop from the TRM paper:
      - T-1 outer cycles WITHOUT gradient (to improve z_L, z_H cheaply)
      - 1 outer cycle WITH gradient (for backpropagation)
      - Each outer cycle = n inner L-cycles + 1 H-cycle

    Uses a SINGLE shared reasoning module (Section 4.3 of the paper).

    Input:  hidden_states (B, T, D) from GPT-2
    Output: refined hidden_states (B, T, D)
    """

    def __init__(self, config: TRMBlockConfig):
        super().__init__()
        self.config = config
        # Single shared reasoning module (key insight from TRM paper: one network suffices)
        self.reasoning = TRMReasoningModule(config)

        # Learnable initial states for z_L and z_H
        self.z_L_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        self.z_H_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)

        # Learnable gating scalar for residual connection.
        # Initialized to -5 so that sigmoid(gate) ≈ 0.007 at the start,
        # meaning the block initially acts as a near-identity pass-through.
        # The model gradually learns to blend in the TRM refinement.
        self.gate = nn.Parameter(torch.tensor(-5.0))

        # Projection to match GPT-2 hidden_size if needed (identity if sizes match)
        self.proj_in = nn.Identity()
        self.proj_out = nn.Identity()

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_embeddings: (B, T, D) hidden states from GPT-2 layers.

        Returns:
            z_H: (B, T, D) refined hidden states to feed into remaining GPT-2 layers.
        """
        B, T, D = input_embeddings.shape

        x = self.proj_in(input_embeddings)

        # Initialize z_H (solution state) and z_L (latent reasoning state)
        z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

        # H_cycles - 1 cycles WITHOUT gradient (cheap forward-only pondering)
        with torch.no_grad():
            for _h in range(self.config.H_cycles - 1):
                for _l in range(self.config.L_cycles):
                    # Update latent reasoning: z_L sees (z_H + input)
                    z_L = self.reasoning(z_L, z_H + x)
                # Update solution: z_H sees z_L
                z_H = self.reasoning(z_H, z_L)

        # 1 final cycle WITH gradient (for training)
        for _l in range(self.config.L_cycles):
            z_L = self.reasoning(z_L, z_H + x)
        z_H = self.reasoning(z_H, z_L)

        # Gated residual: output = α * trm_refined + (1 - α) * input
        # where α = sigmoid(self.gate), starting near 0
        alpha = torch.sigmoid(self.gate)
        return self.proj_out(alpha * z_H + (1.0 - alpha) * x)
