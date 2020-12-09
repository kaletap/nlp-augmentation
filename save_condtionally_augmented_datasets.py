import os

from src.augmentation import augment_dataset, ConditionalMLMInsertionAugmenter, ConditionalMLMSubstitutionAugmenter
from src.pipelines.classification_configs import dataset_configs
from src.pipelines.datasets import get_datasets

# Setup
ROOT_SAVE_DIR = "examples"
CONDITIONAL_MLM_ROOT_PATH = "../mgr-code"
paths = {
    ("ag_news", 1000): "ag-news-conditional-roberta-1000/checkpoint-270",
    ("twitter", 1000): "twitter-conditional-roberta-1000/checkpoint-200",
    ("yelp", 1000): "yelp-conditional-roberta-1000/checkpoint-250"
}
train_size = 1000

augmentation_configs = [
    {
        "name": "conditional_mlm_insertion",
        "class": ConditionalMLMInsertionAugmenter,
        "augmenter_parameters": {
            "min_mask": 1,
            "max_mask": 100,
            "uniform": False
        },
    },
    {
        "name": "conditional_mlm_substitution",
        "class": ConditionalMLMSubstitutionAugmenter,
        "augmenter_parameters": {
            "min_mask": 1,
            "max_mask": 100,
            "uniform": False
        },
    }
]

for name, config in dataset_configs.items():
    if not config["train_sizes"]:
        continue
    for augmentation_config in augmentation_configs:
        print("Dataset:", name, "config:", config)
        print("Augmentation config:", augmentation_config)
        mlm_path = os.path.join(CONDITIONAL_MLM_ROOT_PATH, paths[(name, train_size)])
        print(f"Loading conditional mlm model from {mlm_path}")
        augmenter = augmentation_config["class"](model_name_or_path=mlm_path,
                                                 **augmentation_config["augmenter_parameters"])
        save_dir = os.path.join(ROOT_SAVE_DIR, f"{name}/{augmentation_config['name']}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{name}/{augmentation_config['name']}/", exist_ok=True)
        train_dataset, _, _ = get_datasets(
            config["dataset_name"],
            augmenter=None,
            train_size=train_size,
            val_size=config["val_size"],
            test_size=config["test_size"],
            augmentation_prob=0,
            load_test=config["load_test"],
            text_columns=None,
            # we will need to merge text columns with sep token later on (using DatasetWithMultipleTexts)
            merge_text_columns=False
        )
        n_times = 16 if train_size <= 500 else 8 if train_size <= 2000 else 6
        dataset_df = augment_dataset(train_dataset, augmenter, n_times=n_times, text_columns=config["text_colname"],
                                     label_column=config["label_colname"], label_map=config["label_map"])
        save_path = os.path.join(save_dir, f"{train_size}.csv")
        dataset_df.to_csv(save_path)
        print(f"Saved dataset to {save_path}")
        print(dataset_df.sample(5, random_state=42))
