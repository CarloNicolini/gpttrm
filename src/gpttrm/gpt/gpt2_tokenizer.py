# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer


class GPT2TextEncoder:
    """
    Wrapper around GPT2 tokenizer.
    """

    def __init__(self, pretrained_model) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        # Add padding token
        special_tokens_dict = {"pad_token": "<PAD>"}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"We have added {len(special_tokens_dict)} special tokens")

        # Mapping to old stoi/itos for compatibility
        self.stoi = self.tokenizer.get_vocab()
        self.itos = {v: k for k, v in self.stoi.items()}

    @property
    def unk_index(self) -> int:
        """Returns the index used for the unknown token."""
        return self.tokenizer.unk_token_id

    @property
    def bos_index(self) -> int:
        """Returns the index used for the begin-of-sentence token."""
        return (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )

    @property
    def eos_index(self) -> int:
        """Returns the index used for the end-of-sentence token."""
        return self.tokenizer.eos_token_id

    @property
    def padding_index(self) -> int:
        """Returns the index used for padding."""
        return self.tokenizer.pad_token_id

    @property
    def vocab(self) -> dict:
        """
        Returns:
            dict: Vocabulary of the tokenizer.
        """
        return self.tokenizer.get_vocab()

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int: Number of tokens in the dictionary.
        """
        return len(self.tokenizer)

    def encode(self, sequence: str) -> torch.Tensor:
        """Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.

        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
        """
        vector = self.tokenizer.encode(sequence)
        return torch.tensor(vector)

    def batch_encode(self, iterator, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        :param iterator: Batch of text to encode.
        :param **kwargs: Keyword arguments (not used here, for compatibility).

        Returns:
            torch.Tensor, torch.Tensor: Encoded and padded batch of sequences; Original lengths of sequences.
        """
        encodings = self.tokenizer(
            iterator, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        # Current code expects (tokens, lengths)
        # Calculate lengths manually
        lengths = torch.tensor([len(self.tokenizer.encode(text)) for text in iterator])
        return encodings["input_ids"], lengths

    def decode(self, embeddings):
        """Decodes an encoded sequence.
        :param embeddings: Tensor or list of token ids to decode.

        Returns:
            - str: Decoded string.
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.tolist()
        return self.tokenizer.decode(embeddings, skip_special_tokens=True)
