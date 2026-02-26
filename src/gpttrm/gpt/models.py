"""
GPT-2 + TRM Lightning Modules: Option A and Option B.

Option A – Pre-Head Reasoner:
    Tokens → Full CustomGPT2 → TRM Pondering Block → LM Head

Option B – Interspersed TRM Layer:
    Tokens → GPT2 Layers[0..mid) → TRM Pondering Block → GPT2 Layers[mid..N) → LM Head
"""

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

        # For tracking convergence speed
        self._best_train_loss = float("inf")

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

    # --- Loss & Metrics ---

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

    @torch.no_grad()
    def _compute_token_accuracy(
        self, lm_logits: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Fraction of next-token predictions that are correct (ignoring padding)."""
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        preds = shift_logits.argmax(dim=-1)
        mask = shift_labels != self.tokenizer.padding_index
        correct = ((preds == shift_labels) & mask).sum()
        total = mask.sum().clamp_min(1)
        return correct.float() / total.float()

    def _log_trm_metrics(self, trm_metrics: dict, prefix: str = "train") -> None:
        """Log TRM-specific metrics to TensorBoard."""
        if trm_metrics is None:
            return
        for key, value in trm_metrics.items():
            self.log(
                f"{prefix}/{key}",
                value,
                on_step=(prefix == "train"),
                on_epoch=True,
                prog_bar=(key == "trm/gate_alpha"),
            )

    def _compute_gradient_norms(self) -> dict:
        """Compute gradient norms for GPT-2 and TRM separately."""
        norms = {}
        # GPT-2 gradient norm
        gpt2_grads = [
            p.grad.detach().norm() for p in self.gpt2.parameters() if p.grad is not None
        ]
        if gpt2_grads:
            norms["grad_norm/gpt2"] = torch.stack(gpt2_grads).norm()

        # TRM gradient norm (if TRM block exists)
        if self.trm_block is not None:
            trm_grads = [
                p.grad.detach().norm()
                for p in self.trm_block.parameters()
                if p.grad is not None
            ]
            if trm_grads:
                norms["grad_norm/trm"] = torch.stack(trm_grads).norm()

        # Total gradient norm
        all_grads = [
            p.grad.detach().norm() for p in self.parameters() if p.grad is not None
        ]
        if all_grads:
            norms["grad_norm/total"] = torch.stack(all_grads).norm()

        return norms

    # --- Training / Validation steps ---

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        tokens = batch["tokens"]
        result = self(tokens)

        # Unpack: TRM models return (logits, trm_metrics), baseline returns just logits
        if isinstance(result, tuple):
            logits, trm_metrics = result
        else:
            logits, trm_metrics = result, None

        loss = self.compute_loss(logits, tokens)

        # --- Core metrics ---
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # Also log as train_loss for backward compat
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        # Perplexity (per-step for training)
        with torch.no_grad():
            ppl = torch.exp(loss.clamp(max=20))  # clamp to avoid inf
            self.log("train/perplexity", ppl, on_step=True, on_epoch=True)

        # Token-level accuracy
        acc = self._compute_token_accuracy(logits, tokens)
        self.log("train/token_accuracy", acc, on_step=True, on_epoch=True)

        # --- TRM-specific metrics ---
        self._log_trm_metrics(trm_metrics, prefix="train")

        # --- Convergence speed tracking ---
        with torch.no_grad():
            if loss.item() < self._best_train_loss:
                self._best_train_loss = loss.item()
            self.log(
                "train/best_loss", self._best_train_loss, on_step=True, on_epoch=False
            )

        return loss

    def on_after_backward(self) -> None:
        """Log gradient norms after backward pass (every N steps to save overhead)."""
        if self.global_step % 50 == 0:
            grad_norms = self._compute_gradient_norms()
            for key, value in grad_norms.items():
                self.log(key, value, on_step=True, on_epoch=False)

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        tokens = batch["tokens"]
        result = self(tokens)

        if isinstance(result, tuple):
            logits, trm_metrics = result
        else:
            logits, trm_metrics = result, None

        loss = self.compute_loss(logits, tokens)

        # --- Core metrics ---
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)  # backward compat

        # Token-level accuracy
        acc = self._compute_token_accuracy(logits, tokens)
        self.log("val/token_accuracy", acc, on_epoch=True)

        # --- TRM-specific metrics ---
        self._log_trm_metrics(trm_metrics, prefix="val")

        return loss

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            perplexity = torch.exp(val_loss.clamp(max=20))
            self.log("val/perplexity", perplexity, prog_bar=True)
            self.log("perplexity", perplexity, prog_bar=False)  # backward compat

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
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gpt2 = CustomGPT2Model(self.gpt2_config)
        self.trm_block = TRMBlock(self.trm_config)
        self.lm_head = nn.Linear(
            self.gpt2_config.n_embd, self.gpt2_config.vocab_size, bias=False
        )
        if self.gpt2_config.tie_word_embeddings:
            self.lm_head.weight = self.gpt2.transformer.wte.weight

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
        hidden_states = self.gpt2(idx=tokens)
        refined, trm_metrics = self.trm_block(hidden_states)
        logits = self.lm_head(refined)
        return logits, trm_metrics


# ---------------------------------------------------------------------------
# Option B: Interspersed TRM Layer
# ---------------------------------------------------------------------------


class GPT2TRMOptionB(GPT2TRMBase):
    """
    Option B: TRM inserted mid-way through GPT-2 layers.

    Flow: Tokens → GPT2 Layers[0..mid) → TRM Pondering Block → GPT2 Layers[mid..N) → LM Head
    """

    def __init__(self, trm_insert_layer: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        self.trm_insert_layer = trm_insert_layer or (self.gpt2_config.n_layer // 2)
        self.gpt2 = CustomGPT2Model(self.gpt2_config)
        self.trm_block = TRMBlock(self.trm_config)
        self.lm_head = nn.Linear(
            self.gpt2_config.n_embd, self.gpt2_config.vocab_size, bias=False
        )
        if self.gpt2_config.tie_word_embeddings:
            self.lm_head.weight = self.gpt2.transformer.wte.weight

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # First half of GPT-2
        hidden_states = self.gpt2(
            idx=tokens,
            start_layer=0,
            end_layer=self.trm_insert_layer,
        )
        # TRM pondering at the midpoint
        refined, trm_metrics = self.trm_block(hidden_states)
        # Second half of GPT-2
        hidden_states = self.gpt2(
            inputs_embeds=refined,
            start_layer=self.trm_insert_layer,
            end_layer=self.gpt2_config.n_layer,
        )
        logits = self.lm_head(hidden_states)
        return logits, trm_metrics
