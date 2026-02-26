"""
GPT-2 + TRM Lightning Modules: Option A and Option B.

Option A – Pre-Head Reasoner:
    Tokens → Full CustomGPT2 → TRM Pondering Block → LM Head

Option B – Interspersed TRM Layer:
    Tokens → GPT2 Layers[0..mid) → TRM Pondering Block → GPT2 Layers[mid..N) → LM Head
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from gpttrm.gpt.custom_gpt2 import CustomGPT2Config, CustomGPT2Model
from gpttrm.gpt.trm_block import TRMBlockConfig, TRMBlock
from gpttrm.gpt.gpt2_tokenizer import GPT2TextEncoder
from gpttrm.gpt.dataloader import text_dataset


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------


class GPT2TRMBase(pl.LightningModule):
    """Shared boilerplate for both Option A and Option B."""

    def __init__(
        self,
        gpt2_config: CustomGPT2Config,
        trm_config: TRMBlockConfig,
        learning_rate: float = 3e-4,
        train_csv: str = "data/train_data.csv",
        dev_csv: str = "data/valid_data.csv",
        test_csv: str = "data/valid_data.csv",
        batch_size: int = 8,
        loader_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gpt2_config = gpt2_config
        self.trm_config = trm_config
        self.learning_rate = learning_rate
        self.train_csv = train_csv
        self.dev_csv = dev_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.loader_workers = loader_workers

        # Tokenizer (shared across options)
        self.tokenizer = GPT2TextEncoder("gpt2")
        # Update vocab_size to include the added PAD token
        self.gpt2_config.vocab_size = self.tokenizer.vocab_size

        # Core components (to be populated by subclasses)
        self.gpt2: CustomGPT2Model = None  # type: ignore
        self.trm_block: TRMBlock = None  # type: ignore
        self.lm_head: nn.Linear = None  # type: ignore

        # Loss
        self._loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.padding_index)

    # --- Data ---

    def prepare_sample(self, sample: list) -> dict:
        texts = [s["text"] for s in sample]
        tokens, lengths = self.tokenizer.batch_encode(texts)
        return {"tokens": tokens}

    def train_dataloader(self) -> DataLoader:
        from argparse import Namespace

        hparams = Namespace(
            train_csv=self.train_csv,
            dev_csv=self.dev_csv,
            test_csv=self.test_csv,
        )
        dataset = text_dataset(hparams, val=False, test=False)[0]
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.loader_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        from argparse import Namespace

        hparams = Namespace(
            train_csv=self.train_csv,
            dev_csv=self.dev_csv,
            test_csv=self.test_csv,
        )
        dataset = text_dataset(hparams, train=False, test=False)[0]
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.loader_workers,
        )

    # --- Loss ---

    def compute_loss(
        self, lm_logits: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Standard causal LM loss: predict next token."""
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        return self._loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    # --- Training / Validation steps ---

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        tokens = batch["tokens"]
        logits = self(tokens)
        loss = self.compute_loss(logits, tokens)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        tokens = batch["tokens"]
        logits = self(tokens)
        loss = self.compute_loss(logits, tokens)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            perplexity = torch.exp(val_loss)
            self.log("perplexity", perplexity, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01
        )


# ---------------------------------------------------------------------------
# Option A: Pre-Head Reasoner
# ---------------------------------------------------------------------------


class GPT2TRMOptionA(GPT2TRMBase):
    """
    Option A: TRM as a Post-GPT2 Reasoner.

    Flow: Tokens → Full CustomGPT2 (all layers) → TRM Pondering Block → LM Head

    GPT-2 handles syntax/semantics; TRM adds iterative latent reasoning
    before the final vocabulary projection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Build GPT-2 backbone
        self.gpt2 = CustomGPT2Model(self.gpt2_config)

        # TRM pondering block
        self.trm_block = TRMBlock(self.trm_config)

        # Language model head
        self.lm_head = nn.Linear(
            self.gpt2_config.n_embd, self.gpt2_config.vocab_size, bias=False
        )

        # Optionally tie weights
        if self.gpt2_config.tie_word_embeddings:
            self.lm_head.weight = self.gpt2.transformer.wte.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Full GPT-2 pass
        hidden_states = self.gpt2(idx=tokens)  # (B, T, D)
        # TRM pondering
        refined = self.trm_block(hidden_states)  # (B, T, D)
        # LM head
        logits = self.lm_head(refined)  # (B, T, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Option B: Interspersed TRM Layer
# ---------------------------------------------------------------------------


class GPT2TRMOptionB(GPT2TRMBase):
    """
    Option B: TRM inserted mid-way through GPT-2 layers.

    Flow: Tokens → GPT2 Layers[0..mid) → TRM Pondering Block → GPT2 Layers[mid..N) → LM Head

    This leverages the insight that intermediate layers have the highest
    intrinsic dimension and expressive power, making the TRM reasoning
    maximally effective at this point.
    """

    def __init__(self, trm_insert_layer: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        # Where to insert TRM (default: halfway)
        self.trm_insert_layer = trm_insert_layer or (self.gpt2_config.n_layer // 2)

        # Build GPT-2 backbone
        self.gpt2 = CustomGPT2Model(self.gpt2_config)

        # TRM pondering block
        self.trm_block = TRMBlock(self.trm_config)

        # Language model head
        self.lm_head = nn.Linear(
            self.gpt2_config.n_embd, self.gpt2_config.vocab_size, bias=False
        )

        # Optionally tie weights
        if self.gpt2_config.tie_word_embeddings:
            self.lm_head.weight = self.gpt2.transformer.wte.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # First half of GPT-2
        hidden_states = self.gpt2(
            idx=tokens,
            start_layer=0,
            end_layer=self.trm_insert_layer,
        )  # (B, T, D)

        # TRM pondering at the midpoint
        refined = self.trm_block(hidden_states)  # (B, T, D)

        # Second half of GPT-2
        hidden_states = self.gpt2(
            inputs_embeds=refined,
            start_layer=self.trm_insert_layer,
            end_layer=self.gpt2_config.n_layer,
        )  # (B, T, D)

        # LM head
        logits = self.lm_head(hidden_states)  # (B, T, vocab_size)
        return logits
