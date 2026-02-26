from enum import Enum
from typing import Optional

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import typer

from gpttrm.gpt.custom_gpt2 import CustomGPT2Config
from gpttrm.gpt.trm_block import TRMBlockConfig
from gpttrm.gpt.models import GPT2TRMOptionA, GPT2TRMOptionB, GPT2TRMBase

app = typer.Typer()


class ExperimentType(str, Enum):
    baseline = "baseline"
    option_a = "option_a"
    option_b = "option_b"


class AcceleratorType(str, Enum):
    auto = "auto"
    cpu = "cpu"
    mps = "mps"
    gpu = "gpu"


class GPT2Baseline(GPT2TRMBase):
    """Simple baseline: Standard GPT-2 without TRM reasoning block."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from gpttrm.gpt.custom_gpt2 import CustomGPT2Model
        import torch.nn as nn

        self.gpt2 = CustomGPT2Model(self.gpt2_config)
        self.lm_head = nn.Linear(
            self.gpt2_config.n_embd, self.gpt2_config.vocab_size, bias=False
        )
        if self.gpt2_config.tie_word_embeddings:
            self.lm_head.weight = self.gpt2.transformer.wte.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gpt2(idx=tokens)
        logits = self.lm_head(hidden_states)
        return logits


@app.command()
def train(
    experiment: ExperimentType = typer.Option(
        ExperimentType.baseline, help="Which model variant to train"
    ),
    modern: bool = typer.Option(
        False, help="Whether to use modern architecture (RMSNorm, RoPE, SwiGLU, GQA)"
    ),
    accelerator: AcceleratorType = typer.Option(
        AcceleratorType.auto, help="Device accelerator to use (auto, cpu, mps, gpu)"
    ),
    trm_insert_layer: Optional[int] = typer.Option(
        None, help="For Option B, which layer to insert the TRM block after"
    ),
    batch_size: int = 16,
    max_epochs: int = 5,
    lr: float = 3e-4,
    n_layer: int = 6,
    n_head: int = 4,
    n_embd: int = 128,
    h_cycles: int = 2,
    l_cycles: int = 2,
    n_reasoning_layers: int = 2,
    scheduler: str = typer.Option("wsd", help="Learning rate scheduler: cosine or wsd"),
    warmup_steps: int = 500,
):
    # Setup configs
    gpt2_cfg = CustomGPT2Config(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        modern=modern,
        n_kv_head=n_head // 2 if modern else None,  # Example GQA ratio
    )
    trm_cfg = TRMBlockConfig(
        hidden_size=n_embd,
        n_head=n_head,
        H_cycles=h_cycles,
        L_cycles=l_cycles,
        n_reasoning_layers=n_reasoning_layers,
        modern=modern,
    )

    # Instantiate model
    common_kwargs = dict(
        gpt2_config=gpt2_cfg,
        trm_config=trm_cfg,
        learning_rate=lr,
        lr_scheduler_type=scheduler,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
    )

    if experiment == ExperimentType.baseline:
        model = GPT2Baseline(**common_kwargs)
    elif experiment == ExperimentType.option_a:
        model = GPT2TRMOptionA(**common_kwargs)
    elif experiment == ExperimentType.option_b:
        model = GPT2TRMOptionB(trm_insert_layer=trm_insert_layer, **common_kwargs)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Logger
    exp_suffix = "modern" if modern else "standard"
    logger = TensorBoardLogger("experiments", name=f"{experiment.value}_{exp_suffix}")

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="{epoch}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    # Map AcceleratorType to Lightning string
    acc_val = accelerator.value
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=acc_val,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="16-mixed" if torch.cuda.is_available() else "32",
    )

    # Train
    trainer.fit(model)


@app.command()
def inference(
    checkpoint_path: str = typer.Argument(
        ..., help="Path to the .ckpt model checkpoint"
    ),
    prompt: str = typer.Argument(..., help="Prompt string to start generation"),
    experiment: ExperimentType = typer.Option(
        ExperimentType.baseline, help="Model variant used"
    ),
    modern: bool = typer.Option(
        False, help="Whether the model uses modern architecture"
    ),
    max_new_tokens: int = typer.Option(50, help="Maximum number of tokens to generate"),
    temperature: float = typer.Option(1.0, help="Sampling temperature"),
    top_k: int = typer.Option(50, help="Top-k sampling threshold"),
    n_layer: int = 6,
    n_head: int = 4,
    n_embd: int = 128,
    h_cycles: int = 2,
    l_cycles: int = 2,
    n_reasoning_layers: int = 2,
    trm_insert_layer: Optional[int] = typer.Option(
        None, help="For Option B, which layer to insert the TRM block after"
    ),
):
    """
    Load a model from a checkpoint and generate text from a prompt.
    """
    gpt2_cfg = CustomGPT2Config(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        modern=modern,
        n_kv_head=n_head // 2 if modern else None,
    )
    trm_cfg = TRMBlockConfig(
        hidden_size=n_embd,
        n_head=n_head,
        H_cycles=h_cycles,
        L_cycles=l_cycles,
        n_reasoning_layers=n_reasoning_layers,
        modern=modern,
    )

    common_kwargs = dict(gpt2_config=gpt2_cfg, trm_config=trm_cfg)

    # PyTorch 2.6+ defaults to weights_only=True and blocks custom classes in checkpoints.
    # Our older checkpoints might have saved the configs directly.
    import torch.serialization

    torch.serialization.add_safe_globals([CustomGPT2Config, TRMBlockConfig])

    # Load from checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    if experiment == ExperimentType.baseline:
        model = GPT2Baseline.load_from_checkpoint(checkpoint_path, **common_kwargs)
    elif experiment == ExperimentType.option_a:
        model = GPT2TRMOptionA.load_from_checkpoint(checkpoint_path, **common_kwargs)
    elif experiment == ExperimentType.option_b:
        model = GPT2TRMOptionB.load_from_checkpoint(
            checkpoint_path, trm_insert_layer=trm_insert_layer, **common_kwargs
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    model.eval()

    # Encode prompt
    tokens_tensor, _ = model.tokenizer.batch_encode([prompt])

    # Generate
    print("Generating...")
    output_tokens = model.generate(
        prompt_tokens=tokens_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    # Decode and print
    # Ignore padding index
    output_list = [
        t for t in output_tokens[0].tolist() if t != model.tokenizer.padding_index
    ]
    output_text = model.tokenizer.decode(output_list)
    print("\n" + "=" * 40)
    print("GENERATED TEXT")
    print("=" * 40)
    print(output_text)
    print("=" * 40 + "\n")


if __name__ == "__main__":
    app()
