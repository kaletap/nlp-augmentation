import json
import os
import shutil
import warnings
from datetime import datetime

import pandas as pd
import torch

from src.data_processing import TokenizedDataCollator, Tokenizer
from src.model import CnnClassifier, CnnClassifierConfig
from src.pipelines.classification_configs import (
    augmentation_configs,
    cnn_classifier_config,
    dataset_configs,
    trainer_config_dict,
    root_mlm_dir,
    root_output_dir,
    root_training_csv_path
)
from src.pipelines.datasets import get_datasets,  DatasetWithTokenization
from src.training_utils import Trainer, TrainerConfig


PLATFORM = "local"

# Setup
ROOT_OUTPUT_DIR = root_output_dir[PLATFORM]
ROOT_MLM_DIR = root_mlm_dir[PLATFORM]
ROOT_TRAINING_CSV_PATH = root_training_csv_path[PLATFORM]
SAVE_DIR = "."


def run_exp():
    results_df = pd.DataFrame(columns=["dataset", "augmentation", "training_size", "vocab_words", "seq_len", "accuracy"])
    for name, config in dataset_configs.items():  # name is a name of dataset
        if not config["train_sizes"]:
            continue
        for augmentation_config in augmentation_configs:
            print("Dataset:", name, "config:", config)
            print("Augmentation config:", augmentation_config)
            augmentation_name = augmentation_config["name"]
            mlm_relative_path = config.get("mlm_relative_path", None)
            use_finetuned = augmentation_config.get("use_finetuned", None)
            if use_finetuned and not mlm_relative_path:
                warnings.warn(f"You are asking to use finetuned model for dataset {name} but do not provide path to the"
                              f"pretrained model")
            if ROOT_TRAINING_CSV_PATH:
                augmenter = None
            else:  # we only create augmenter if we do not use already augmented dataset
                if use_finetuned and mlm_relative_path:
                    mlm_path = os.path.join(ROOT_MLM_DIR, mlm_relative_path)
                    print(f"Loading model for augmentation from {mlm_path}")
                    augmenter = augmentation_config["class"](model_name_or_path=mlm_path, **augmentation_config["augmenter_parameters"])
                else:
                    augmenter = augmentation_config["class"](**augmentation_config["augmenter_parameters"])
            for train_size in config["train_sizes"]:
                training_csv_path = os.path.join(ROOT_TRAINING_CSV_PATH, name, augmentation_name, f"{train_size}.csv")
                train_dataset, val_dataset, test_dataset = get_datasets(
                    config["dataset_name"],
                    augmenter=augmenter,
                    train_size=train_size,
                    val_size=config["val_size"],
                    test_size=config["test_size"],
                    augmentation_prob=augmentation_config["augmentation_prob"],
                    load_test=config["load_test"],
                    text_columns=config["text_colname"],
                    sep_token=Tokenizer.sep_token,
                    training_csv_path=training_csv_path
                )

                tokenizer = Tokenizer.from_dataset(train_dataset, num_words=10_000, min_occ=1)  # creating tokenizer
                print(f"Created tokenizer with {len(tokenizer)} vocab words. Using sequence length of {tokenizer.max_length}")
                train_dataset = DatasetWithTokenization(train_dataset, tokenizer)
                val_dataset = DatasetWithTokenization(val_dataset, tokenizer)
                test_dataset = DatasetWithTokenization(test_dataset, tokenizer)

                data_collator = TokenizedDataCollator(label_colname=config["label_colname"],
                                                      pad_token_id=tokenizer.pad_token_id, padding=tokenizer.max_length)
                model_config = CnnClassifierConfig(num_labels=config["num_labels"], seq_len=tokenizer.max_length,
                                                   num_words=len(tokenizer), **cnn_classifier_config)
                model = CnnClassifier(model_config)

                print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print(f"Training dataset size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
                print(train_dataset[0])
                print(val_dataset[0])
                print(test_dataset[0])

                num_train_epochs = {20: 5, 100: 4}.get(train_size, 3)

                output_dir = os.path.join(ROOT_OUTPUT_DIR, "cnn_model.pt")
                trainer_config = TrainerConfig(ckpt_path=output_dir, max_epochs=num_train_epochs, **trainer_config_dict)
                trainer = Trainer(
                    model=model,
                    train_dataset=train_dataset,
                    test_dataset=val_dataset,
                    collator=data_collator,
                    config=trainer_config,
                    compute_metrics=config["metrics_function"]
                )

                trainer.train()

                # Loading the best model
                model.load_state_dict(torch.load(output_dir))
                test_result = trainer.evaluate(test_dataset)
                print(test_result)
                results_df = results_df.append({
                    "dataset": name,
                    "augmentation": augmentation_config["name"],
                    "training_size": train_size,
                    "vocab_words": len(tokenizer),
                    "seq_len": tokenizer.max_length,
                    "accuracy": test_result["accuracy"]
                }, ignore_index=True)
                results_df.to_csv(os.path.join(SAVE_DIR, 'results.csv'), index=False)
                print(results_df)
                shutil.rmtree(output_dir, ignore_errors=True)
                print()


if __name__ == "__main__":
    run_exp()
