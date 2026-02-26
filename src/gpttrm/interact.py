# -*- coding: utf-8 -*-
import argparse

from .gpt2_lm import GPT2LanguageModel


def load_model_from_checkpoint(checkpoint_path: str):
    """
    Function that loads the model from a checkpoint.
    :param checkpoint_path: Path to the checkpoint file.
    Return:
        - Pretrained model.
    """
    print(f"Loading model from {checkpoint_path}...")
    model = GPT2LanguageModel.load_from_checkpoint(checkpoint_path)
    # Make sure model is in prediction mode
    model.eval()
    model.freeze()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimalist GPT2 Generator")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to the checkpoint file.",
    )
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.checkpoint)

    while True:
        print(
            "\nPlease write some text or quit to exit the interactive shell ('quit' or 'q'):"
        )
        # Get input sentence
        input_sentence = input("> ")
        if input_sentence.lower() in ["q", "quit"]:
            break
        if not input_sentence.strip():
            continue

        generated_text = model.generate(sample={"text": input_sentence})
        print(f"\nGenerated: {generated_text}")
