import pandas as pd
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_lists(text: list) -> list:
    """Converts each line into a list of dictionaries."""
    collated_dataset = []
    for i in range(len(text)):
        collated_dataset.append({"text": str(text[i])})
    return collated_dataset


def text_dataset(hparams, train=True, val=True, test=True):
    """
    Loads the Dataset from the csv files passed to the hparams.
    :param hparams: Namespace object containing the path to the data files.
    :param train: flag to return the train set.
    :param val: flag to return the validation set.
    :param test: flag to return the test set.

    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """

    def load_dataset(path):
        df = pd.read_csv(path)
        text = list(df.text)
        return TextDataset(collate_lists(text))

    func_out = []
    if train:
        func_out.append(load_dataset(hparams.train_csv))
    if val:
        func_out.append(load_dataset(hparams.dev_csv))
    if test:
        func_out.append(load_dataset(hparams.test_csv))

    return tuple(func_out)
