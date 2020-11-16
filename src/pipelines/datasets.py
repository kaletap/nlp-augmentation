import random

from datasets import load_dataset
from torch.utils.data import Dataset


class DatasetWithAugmentation(Dataset):
    def __init__(self, dataset, augmenter, augmentation_prob: float = 0.7):
        self.dataset = dataset
        self.augmenter = augmenter
        self.augmentation_prob = augmentation_prob
        self.n_errors = 0
        self.max_errors = 50

    def __getitem__(self, i):
        item = self.dataset[i]
        if random.random() < self.augmentation_prob:
            try:
                item['text'] = self.augmenter(item['text'])
            except Exception as e:
                print(f"Something went wrong when augmenting item number {i}: {e}")
                print(item)
                if self.n_errors > self.max_errors:
                    raise Exception(f"Number of error exceeded {self.max_errors}!")
        return item

    def __len__(self):
        return len(self.dataset)


class DatasetWithMultipleTexts(Dataset):
    def __init__(self, dataset, text_columns, sep_token):
        self.dataset = dataset
        self.text_columns = text_columns
        self.sep_token = sep_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        text = f' {self.sep_token} '.join([item[col] for col in self.text_columns])
        item['text'] = text
        return item


def get_datasets(dataset_name, augmenter=None, train_size=10_000, val_size=5_000, test_size=None, augmentation_prob=0.7,
                 random_seed: int = 42, load_test=False, filter_func=None, text_columns=None, sep_token=None):
    """
    Returns a tuple of train, validation and test datasets of sizes determined by arguments.
    If load_test is False, test_size has to be specified.
    Random seeds are set in order to get the same validation and test sets in every experiment."
    """
    dataset = load_dataset(dataset_name, split="train")
    # We want test and validation data to be the same for every experiment
    assert test_size is not None or load_test, "Cannot load test dataset with load_test=False when test_size is None"
    if load_test:
        test_dataset = load_dataset(dataset_name, split="test")
        if test_size:
            test_dataset = test_dataset.train_test_split(test_size=test_size, seed=random_seed)["test"]
    else:
        split = dataset.train_test_split(test_size=test_size, seed=random_seed)
        test_dataset = split["test"]
        dataset = split["train"]
    if dataset_name == "snli":
        filter_func = lambda d: d['label'] != -1
    if filter_func:
        dataset = dataset.filter(filter_func)
        test_dataset = test_dataset.filter(filter_func)
    train_val_split = dataset.train_test_split(test_size=val_size, seed=random_seed)
    # we only want to use train_size samples for training
    train_dataset = train_val_split["train"].train_test_split(train_size=train_size, seed=random_seed)["train"]
    val_dataset = train_val_split["test"]
    if text_columns and type(text_columns) == list:
        assert sep_token is not None, "Sep token has to be specified when using multiple text columns"
        train_dataset = DatasetWithMultipleTexts(train_dataset, text_columns, sep_token)
        val_dataset = DatasetWithMultipleTexts(val_dataset, text_columns, sep_token)
        test_dataset = DatasetWithMultipleTexts(test_dataset, text_columns, sep_token)
    if augmenter:
        train_dataset = DatasetWithAugmentation(train_dataset, augmenter, augmentation_prob=augmentation_prob)
    return train_dataset, val_dataset, test_dataset
