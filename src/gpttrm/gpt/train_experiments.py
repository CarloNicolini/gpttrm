"""
Unified training CLI for GPT-2 + TRM experiments.

Usage:
    # Option A (TRM as post-GPT2 reasoner):
    gpttrm-cli train --experiment option-a

    # Option B (TRM interspersed mid-layer):
    gpttrm-cli train --experiment option-b

    # Baseline (vanilla custom GPT-2, no TRM):
    gpttrm-cli train --experiment baseline

All experiments log to TensorBoard under experiments/<experiment_name>.
Designed for Apple Silicon MPS with reduced model dimensions.
"""

import os
from enum import Enum
from typing import Optional

import torch
import typer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from gpttrm.gpt.custom_gpt2 import CustomGPT2Config
from gpttrm.gpt.trm_block import TRMBlockConfig
from gpttrm.gpt.models import GPT2TRMOptionA, GPT2TRMOptionB

app = typer.Typer(
    name="gpttrm-cli",
    help="GPT-2 + TRM Training Experiments CLI",
    add_completion=False,
)


class ExperimentType(str, Enum):
    baseline = "baseline"
    option_a = "option-a"
    option_b = "option-b"


# ---------------------------------------------------------------------------
# Baseline: Vanilla GPT-2 (no TRM) for comparison
# ---------------------------------------------------------------------------


class GPT2Baseline(GPT2TRMOptionA):
    """Vanilla custom GPT-2 baseline without the TRM block."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gpt2(idx=tokens)
        logits = self.lm_head(hidden_states)
        return logits


def _build_model(
    experiment: ExperimentType, gpt2_config, trm_config, common_kwargs, trm_insert_layer
):
    if experiment == ExperimentType.baseline:
        return GPT2Baseline(**common_kwargs)
    elif experiment == ExperimentType.option_a:
        return GPT2TRMOptionA(**common_kwargs)
    elif experiment == ExperimentType.option_b:
        return GPT2TRMOptionB(trm_insert_layer=trm_insert_layer, **common_kwargs)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


@app.callback(invoke_without_command=True)
def train(
    experiment: ExperimentType = typer.Option(
        ExperimentType.baseline, help="Which experiment to run."
    ),
    # Training
    seed: int = typer.Option(42, help="Random seed."),
    max_epochs: int = typer.Option(1, help="Maximum number of training epochs."),
    min_epochs: int = typer.Option(1, help="Minimum number of training epochs."),
    patience: int = typer.Option(5, help="Early stopping patience."),
    learning_rate: float = typer.Option(3e-4, help="Learning rate."),
    batch_size: int = typer.Option(32, help="Batch size."),
    accumulate_grad_batches: int = typer.Option(1, help="Gradient accumulation steps."),
    loader_workers: int = typer.Option(
        0, help="DataLoader workers (0 is safest for MPS)."
    ),
    # GPT-2 architecture
    n_embd: int = typer.Option(256, help="GPT-2 embedding dimension."),
    n_head: int = typer.Option(4, help="GPT-2 attention heads."),
    n_layer: int = typer.Option(6, help="GPT-2 transformer layers."),
    seq_len: int = typer.Option(256, help="Maximum sequence length."),
    # TRM architecture
    trm_l_cycles: int = typer.Option(
        4, help="Inner recursion steps per full TRM cycle."
    ),
    trm_h_cycles: int = typer.Option(2, help="Outer supervision cycles."),
    trm_n_reasoning_layers: int = typer.Option(
        2, help="Layers in TRM reasoning module."
    ),
    trm_insert_layer: Optional[int] = typer.Option(
        None, help="Option B: GPT-2 layer to insert TRM at (default: n_layer//2)."
    ),
    # Data
    train_csv: str = typer.Option("data/train_data.csv", help="Training CSV path."),
    dev_csv: str = typer.Option("data/valid_data.csv", help="Validation CSV path."),
    test_csv: str = typer.Option("data/valid_data.csv", help="Test CSV path."),
):
    """Train a GPT-2 model, optionally augmented with a TRM reasoning block."""
    pl.seed_everything(seed)

    gpt2_config = CustomGPT2Config(
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        seq_len=seq_len,
    )
    trm_config = TRMBlockConfig(
        hidden_size=n_embd,
        n_head=n_head,
        L_cycles=trm_l_cycles,
        H_cycles=trm_h_cycles,
        n_reasoning_layers=trm_n_reasoning_layers,
    )
    common_kwargs = dict(
        gpt2_config=gpt2_config,
        trm_config=trm_config,
        learning_rate=learning_rate,
        train_csv=train_csv,
        dev_csv=dev_csv,
        test_csv=test_csv,
        batch_size=batch_size,
        loader_workers=loader_workers,
    )

    model = _build_model(
        experiment, gpt2_config, trm_config, common_kwargs, trm_insert_layer
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Experiment: {experiment.value}")
    typer.echo(f"Total parameters: {total_params:,}")
    typer.echo(f"Trainable parameters: {trainable_params:,}")
    typer.echo(
        f"GPT-2: n_layer={n_layer}, n_embd={n_embd}, n_head={n_head}, seq_len={seq_len}"
    )
    typer.echo(
        f"TRM: L_cycles={trm_l_cycles}, H_cycles={trm_h_cycles}, "
        f"n_reasoning_layers={trm_n_reasoning_layers}"
    )
    typer.echo(
        f"Training: batch_size={batch_size}, max_epochs={max_epochs}, lr={learning_rate}"
    )
    typer.echo(f"{'=' * 60}\n")

    logger = TensorBoardLogger("experiments/", name=experiment.value)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, verbose=True, mode="min"),
        ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            filename="{epoch}-{val_loss:.4f}-{perplexity:.2f}",
            save_top_k=2,
            verbose=True,
            monitor="val_loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if torch.backends.mps.is_available():
        accelerator, devices = "mps", 1
    elif torch.cuda.is_available():
        accelerator, devices = "gpu", 1
    else:
        accelerator, devices = "cpu", "auto"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        precision="32-true",
    )
    trainer.fit(model)


if __name__ == "__main__":
    app()
