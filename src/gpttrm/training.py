# -*- coding: utf-8 -*-
import argparse
import os

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from gpttrm.gpt2_lm import GPT2LanguageModel


def main(hparams) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """
    pl.seed_everything(hparams.seed)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GPT2LanguageModel(hparams)

    # ------------------------
    # 2 INIT LOGGERS
    # ------------------------
    logger = TensorBoardLogger("experiments/", name="gpt2_training")

    # ------------------------
    # 3 INIT CALLBACKS
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="{epoch}-{val_loss:.2f}-{perplexity:.2f}",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        mode=hparams.metric_mode,
    )

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        accelerator="auto",
        devices=hparams.gpus if hparams.gpus > 0 else "auto",
        strategy="auto",
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        precision=16 if hparams.use_16bit else 32,
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    parser = argparse.ArgumentParser(description="Minimalist GPT2 Generator")

    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="perplexity", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="min",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        help="Number of epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=10,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=2,
        type=int,
        help="Accumulated gradients runs K small batches before doing a backwards pass.",
    )

    # gpu args
    parser.add_argument(
        "--gpus", type=int, default=0, help="How many gpus (0 for auto)"
    )
    parser.add_argument(
        "--use_16bit",
        action="store_true",
        help="If true uses 16 bit precision",
    )

    # each LightningModule defines arguments relevant to it
    parser = GPT2LanguageModel.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
