import os

from src.augmentation import augment_dataset
from src.pipelines.classification_configs import augmentation_configs, dataset_configs
from src.pipelines.datasets import get_datasets

# Setup
ROOT_SAVE_DIR = "examples"
MLM_ROOT_PATH = "../mgr-code/"  # "/content/drive/MyDrive/Colab Notebooks/nlp/pretrained_models"

for name, config in dataset_configs.items():
    if not config["train_sizes"]:
        continue
    for augmentation_config in augmentation_configs:
        print("Dataset:", name, "config:", config)
        print("Augmentation config:", augmentation_config)
        mlm_relative_path = config.get("mlm_relative_path", None)
        use_finetuned = augmentation_config.get("use_finetuned", None)
        if use_finetuned and not mlm_relative_path:
            raise Exception(f"You are asking to use finetuned model for dataset {name} but do not provide path to the "
                            f"pretrained model")
        if use_finetuned and mlm_relative_path:
            mlm_path = os.path.join(MLM_ROOT_PATH, mlm_relative_path)
            print(f"Loading model for augmentation from {mlm_path}")
            augmenter = augmentation_config["class"](model_name_or_path=mlm_path,
                                                     **augmentation_config["augmenter_parameters"])
        else:
            augmenter = augmentation_config["class"](**augmentation_config["augmenter_parameters"])
        save_dir = os.path.join(ROOT_SAVE_DIR, f"{name}/{augmentation_config['name']}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{name}/{augmentation_config['name']}/", exist_ok=True)
        for train_size in config["train_sizes"]:
            train_dataset, val_dataset, test_dataset = get_datasets(
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
            dataset_df = augment_dataset(train_dataset, augmenter, n_times=n_times, text_columns=config["text_colname"])
            save_path = os.path.join(save_dir, f"{train_size}.csv")
            dataset_df.to_csv(save_path)
            print(f"Saved dataset to {save_path}")
            print(dataset_df.sample(5, random_state=42))
