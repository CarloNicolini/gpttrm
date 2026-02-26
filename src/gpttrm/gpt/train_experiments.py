"""
Unified training script for GPT-2 + TRM experiments.

Usage:
    # Option A (TRM as post-GPT2 reasoner):
    python -m gpttrm.gpt.train_experiments --experiment option_a

    # Option B (TRM interspersed mid-layer):
    python -m gpttrm.gpt.train_experiments --experiment option_b

    # Baseline (vanilla custom GPT-2, no TRM):
    python -m gpttrm.gpt.train_experiments --experiment baseline

All experiments log to TensorBoard under experiments/<experiment_name>.
Designed for Apple Silicon MPS with reduced model dimensions.
"""

import argparse
import os

import torch
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


# ---------------------------------------------------------------------------
# Baseline: Vanilla GPT-2 (no TRM) for comparison
# ---------------------------------------------------------------------------


class GPT2Baseline(GPT2TRMOptionA):
    """Vanilla custom GPT-2 baseline without the TRM block.
    Inherits from OptionA but skips TRM in forward pass.
    """

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gpt2(idx=tokens)
        logits = self.lm_head(hidden_states)
        return logits


# ---------------------------------------------------------------------------
# Default configurations tuned for Apple Silicon MPS (≤16 GB unified memory)
# ---------------------------------------------------------------------------


def make_small_gpt2_config(**overrides) -> CustomGPT2Config:
    """Small GPT-2 config for MPS training.
    ~6M params vs 124M for GPT-2 small.
    """
    defaults = dict(
        vocab_size=50257,  # Will be updated by tokenizer
        seq_len=256,  # Shorter sequences for faster iteration
        n_embd=256,  # Reduced hidden dimension
        n_head=4,  # Fewer heads
        n_layer=6,  # Fewer layers
        dropout=0.1,
        tie_word_embeddings=True,
    )
    defaults.update(overrides)
    return CustomGPT2Config(**defaults)


def make_trm_config(**overrides) -> TRMBlockConfig:
    """Small TRM config matching the GPT-2 hidden dimension."""
    defaults = dict(
        hidden_size=256,  # Must match n_embd
        n_head=4,
        expansion=2.0,
        n_reasoning_layers=2,  # The "tiny" part: only 2 layers
        L_cycles=4,  # Inner recursion steps
        H_cycles=2,  # Outer supervision cycles
        dropout=0.1,
    )
    defaults.update(overrides)
    return TRMBlockConfig(**defaults)


def build_model(
    experiment: str, gpt2_config: CustomGPT2Config, trm_config: TRMBlockConfig, args
):
    """Build the appropriate Lightning model for the chosen experiment."""
    common_kwargs = dict(
        gpt2_config=gpt2_config,
        trm_config=trm_config,
        learning_rate=args.learning_rate,
        train_csv=args.train_csv,
        dev_csv=args.dev_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        loader_workers=args.loader_workers,
    )

    if experiment == "baseline":
        return GPT2Baseline(**common_kwargs)
    elif experiment == "option_a":
        return GPT2TRMOptionA(**common_kwargs)
    elif experiment == "option_b":
        return GPT2TRMOptionB(
            trm_insert_layer=args.trm_insert_layer,
            **common_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown experiment: {experiment}. Choose from: baseline, option_a, option_b"
        )


def main(args):
    pl.seed_everything(args.seed)

    # Build configs
    gpt2_config = make_small_gpt2_config(
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        seq_len=args.seq_len,
    )
    trm_config = make_trm_config(
        hidden_size=args.n_embd,  # Must match GPT-2
        n_head=args.n_head,
        L_cycles=args.trm_l_cycles,
        H_cycles=args.trm_h_cycles,
        n_reasoning_layers=args.trm_n_reasoning_layers,
    )

    # Build model
    model = build_model(args.experiment, gpt2_config, trm_config, args)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 60}")
    print(f"Experiment: {args.experiment}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(
        f"GPT-2 config: n_layer={gpt2_config.n_layer}, n_embd={gpt2_config.n_embd}, n_head={gpt2_config.n_head}"
    )
    print(
        f"TRM config: L_cycles={trm_config.L_cycles}, H_cycles={trm_config.H_cycles}, "
        f"n_reasoning_layers={trm_config.n_reasoning_layers}"
    )
    print(f"{'=' * 60}\n")

    # Logger
    logger = TensorBoardLogger("experiments/", name=args.experiment)

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=args.patience,
            verbose=True,
            mode="min",
        ),
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

    # Determine accelerator
    if torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    elif torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    else:
        accelerator = "cpu"
        devices = "auto"

    # Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        precision="32-true",  # MPS works best with float32 for now
    )

    # Train
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPT-2 + TRM Training Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment selection
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=["baseline", "option_a", "option_b"],
        help="Which experiment to run.",
    )

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument(
        "--loader_workers",
        type=int,
        default=0,
        help="DataLoader workers. 0 is safest for MPS.",
    )

    # GPT-2 architecture
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--seq_len", type=int, default=256)

    # TRM architecture
    parser.add_argument(
        "--trm_l_cycles",
        type=int,
        default=4,
        help="Inner recursion steps per full TRM cycle.",
    )
    parser.add_argument(
        "--trm_h_cycles",
        type=int,
        default=2,
        help="Outer supervision cycles (gradient-free warmup + 1 with grad).",
    )
    parser.add_argument(
        "--trm_n_reasoning_layers",
        type=int,
        default=2,
        help="Number of layers in the TRM reasoning module.",
    )
    parser.add_argument(
        "--trm_insert_layer",
        type=int,
        default=None,
        help="For Option B: which GPT-2 layer to insert TRM at (default: n_layer//2).",
    )

    # Data
    parser.add_argument("--train_csv", type=str, default="data/train_data.csv")
    parser.add_argument("--dev_csv", type=str, default="data/valid_data.csv")
    parser.add_argument("--test_csv", type=str, default="data/valid_data.csv")

    args = parser.parse_args()
    main(args)
