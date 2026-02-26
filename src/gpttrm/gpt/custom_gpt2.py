import math
from dataclasses import dataclass
from typing import Optional, Tuple

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


class CustomGPT2Attention(nn.Module):
    def __init__(self, config: CustomGPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.seq_len, config.seq_len)).view(
                1, 1, config.seq_len, config.seq_len
            ),
        )

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Flash attention natively supported in PyTorch >= 2.0 via scaled_dot_product_attention
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CustomGPT2MLP(nn.Module):
    def __init__(self, config: CustomGPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CustomGPT2Block(nn.Module):
    def __init__(self, config: CustomGPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CustomGPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
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
                wpe=nn.Embedding(config.seq_len, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [CustomGPT2Block(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
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
        """
        Allows passing either tokens (idx) or raw floating embeddings (inputs_embeds).
        Also allows running a subset of layers from [start_layer : end_layer).
        This makes inserting mid-layer "pondering" blocks trivial.
        """
        device = idx.device if idx is not None else inputs_embeds.device

        if (idx is not None) and (inputs_embeds is None):
            b, t = idx.size()
            assert t <= self.config.seq_len, (
                f"Cannot forward sequence of length {t}, block size is only {self.config.seq_len}"
            )
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            # forward the GPT model itself
            tok_emb = self.transformer.wte(
                idx
            )  # token embeddings of shape (b, t, n_embd)
            pos_emb = self.transformer.wpe(
                pos
            )  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        elif (inputs_embeds is not None) and (idx is None):
            # If resuming mid-way, inputs_embeds is literally just the hidden state x being passed back in.
            # No positional encodings are added again since x should already include them if start_layer > 0.
            # If start_layer == 0 but we passed soft prompts, we should add pos embeddings.
            # For simplicity: if start_layer > 0, we assume x is an intermediate hidden state.
            if start_layer == 0:
                b, t, _ = inputs_embeds.size()
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                pos_emb = self.transformer.wpe(pos)
                x = self.transformer.drop(inputs_embeds + pos_emb)
            else:
                x = inputs_embeds
        else:
            raise ValueError("Pass either idx or inputs_embeds, not both.")

        # run the transformer blocks
        if end_layer is None:
            end_layer = self.config.n_layer

        for block in self.transformer.h[start_layer:end_layer]:
            x = block(x)

        # Only layer-norm if we are at the very end
        if end_layer == self.config.n_layer:
            x = self.transformer.ln_f(x)

        return x
