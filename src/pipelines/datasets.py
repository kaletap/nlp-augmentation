import random

from datasets import load_dataset
from torch.utils.data import Dataset


class DatasetWithAugmentation(Dataset):
    def __init__(self, dataset, augmenter, augmentation_prob: float = 0.7):
        self.dataset = dataset
        self.augmenter = augmenter
        self.augmentation_prob = augmentation_prob

    def __getitem__(self, i):
        item = self.dataset[i]
        if random.random() < self.augmentation_prob:
            try:
                item['text'] = self.augmenter(item['text'])
            except:
                print(f"Something went wrong when augmenting item number {i}")
                print(item)
                return item
        else:
            return item

    def __len__(self):
        return len(self.dataset)


def get_datasets(dataset_name, augmenter=None, train_size=10_000, val_size=5_000, test_size=None, augmentation_prob=0.7,
                 random_seed: int = 42, load_test=False, filter_func=None):
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
        test_dataset = dataset.filter(filter_func)
    train_val_split = dataset.train_test_split(test_size=val_size, seed=random_seed)
    # we only want to use train_size samples for training
    train_dataset = train_val_split["train"].train_test_split(train_size=train_size, seed=random_seed)["train"]
    if augmenter:
        train_dataset = DatasetWithAugmentation(train_dataset, augmenter, augmentation_prob=augmentation_prob)
    val_dataset = train_val_split["test"]
    return train_dataset, val_dataset, test_dataset