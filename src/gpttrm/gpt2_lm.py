import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoModel
import lightning.pytorch as pl

from .dataloader import text_dataset
from .utils import top_k_top_p_filtering, load_weights_lm_head
from .gpt2_tokenizer import GPT2TextEncoder


class GPT2LanguageModel(pl.LightningModule):
    """
    Sample model to show how to train GPT2 with a Language Model head.

    :param hparams: Namespace containing the hyperparameters.
    """

    def __init__(self, hparams) -> None:
        super(GPT2LanguageModel, self).__init__()
        if isinstance(hparams, dict):
            from argparse import Namespace

            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.output_units = 768
        # build model
        self.__build_model()
        # Loss criterion initialization.
        self.__build_loss()

    def __build_model(self) -> None:
        """Init GPT2 model + tokenizer + language model head."""
        self.gpt2 = AutoModel.from_pretrained("gpt2", output_hidden_states=True)
        # Tokenizer
        self.tokenizer = GPT2TextEncoder("gpt2")

        # Resize embeddings to include the added tokens
        self.gpt2.resize_token_embeddings(len(self.tokenizer.tokenizer))

        self.lm_head = nn.Linear(self.output_units, self.gpt2.vocab_size, bias=False)

        # Transfer weights from the GPT2LMHeadModel to our LM Head and add dimensions for added tokens
        original_model = GPT2LMHeadModel.from_pretrained("gpt2")
        load_weights_lm_head(self, original_model)

    def __build_loss(self):
        """Initializes the loss function/s."""
        self._loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.padding_index)

    def generate(self, sample: dict) -> str:
        """Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            input_seq = sample["text"]
            inputs = self.tokenizer.encode(input_seq)
            bos_tokens = torch.full(
                [1], self.tokenizer.stoi["<|endoftext|>"], dtype=torch.long
            )
            shifted_input = torch.cat((bos_tokens, inputs))

            output_seq = shifted_input[: len(inputs) + 1]
            predicted_token = torch.Tensor([0])
            while (
                predicted_token.item() != self.tokenizer.padding_index
                and len(output_seq) < 50
            ):
                outputs = self.forward(output_seq.unsqueeze(0))
                lm_logits = outputs["lm_logits"][0]
                logits = lm_logits[-1, :]
                top_k, top_p, temperature = 0, 0.95, 1
                filtered_logits = top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p, temperature=temperature
                )
                probabilities = F.softmax(filtered_logits, dim=-1)
                predicted_token = torch.multinomial(probabilities, 1)
                output_seq = torch.cat([output_seq, predicted_token])
                if predicted_token.item() == self.tokenizer.tokenizer.eos_token_id:
                    break

            output_sentence = self.tokenizer.decode(output_seq)
            print(output_sentence)

        return output_sentence

    def forward(self, tokens):
        """Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        gpt2_outputs = self.gpt2(tokens)
        hidden_states = gpt2_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        return {"lm_logits": lm_logits}

    def loss(self, predictions: dict, labels: dict) -> torch.Tensor:
        """
        Computes Causal Language Modelling (CLM) Loss value.
        """
        batch_logits = predictions["lm_logits"][..., :-1, :].contiguous()
        target_labels = labels["tokens"][..., 1:].contiguous()
        loss = self._loss(
            batch_logits.view(-1, batch_logits.size(-1)), target_labels.view(-1)
        )
        return loss

    def prepare_sample(self, sample: list) -> dict:
        """
        Function that prepares a sample to input the model.
        """
        texts = [s["text"] for s in sample]
        tokens, lengths = self.tokenizer.batch_encode(texts)
        return {"tokens": tokens}

    def training_step(self, batch: dict, batch_nb: int) -> torch.Tensor:
        """
        Runs one training step.
        """
        model_out = self.forward(batch["tokens"])
        loss_val = self.loss(model_out, batch)

        self.log("train_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True)
        return loss_val

    def validation_step(self, batch: dict, batch_nb: int) -> torch.Tensor:
        """
        Similar to the training step but with the model in eval mode.
        """
        model_out = self.forward(batch["tokens"])
        loss_val = self.loss(model_out, batch)

        self.log("val_loss", loss_val, on_epoch=True, prog_bar=True)
        return loss_val

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of validation to log metrics.
        """
        # Logic for perplexity if needed, but self.log(..., on_epoch=True)
        # already gives us the average val_loss.
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            perplexity = torch.exp(val_loss)
            self.log("perplexity", perplexity, prog_bar=True)

    def configure_optimizers(self):
        """Sets Learning rate for different parameter groups."""
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        """Function that loads the train set."""
        dataset = text_dataset(self.hparams, val=False, test=False)[0]
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        dataset = text_dataset(self.hparams, train=False, test=False)[0]
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Function that loads the test set."""
        dataset = text_dataset(self.hparams, train=False, val=False)[0]
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @classmethod
    def add_model_specific_args(cls, parser):
        """Parser for Estimator specific arguments/hyperparameters."""
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Learning rate.",
        )
        parser.add_argument(
            "--train_csv",
            default="data/train_data.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="data/valid_data.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="data/valid_data.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=4,
            type=int,
            help="How many subprocesses to use for data loading.",
        )
        return parser
