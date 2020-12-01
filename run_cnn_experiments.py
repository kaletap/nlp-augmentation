import json
import os
import shutil
import warnings
from collections import defaultdict
from datetime import datetime

import torch

from src.data_processing import TokenizedDataCollator, Tokenizer
from src.model import CnnClassifier, CnnClassifierConfig
from src.pipelines.classification_configs import augmentation_configs, cnn_classifier_config, dataset_configs, trainer_config_dict
from src.pipelines.datasets import get_datasets,  DatasetWithTokenization
from src.training_utils import Trainer, TrainerConfig


# Setup
ROOT_OUTPUT_DIR = "/kaggle/working"  # "/content/drive/MyDrive/Colab Notebooks/nlp/results"
SAVE_DIR = "."  # ROOT_OUTPUT_DIR
MLM_ROOT_PATH = "/kaggle/input/"  # "/content/drive/MyDrive/Colab Notebooks/nlp/pretrained_models"


def run_exp():
    accuracies = defaultdict(list)
    for name, config in dataset_configs.items():
        if not config["train_sizes"]:
            continue
        for augmentation_config in augmentation_configs:
            print("Dataset:", name, "config:", config)
            print("Augmentation config:", augmentation_config)
            mlm_relative_path = config.get("mlm_relative_path", None)
            use_finetuned = augmentation_config.get("use_finetuned", None)
            if use_finetuned and not mlm_relative_path:
                warnings.warn(f"You are asking to use finetuned model for dataset {name} but do not provide path to the"
                              f"pretrained model")
            if use_finetuned and mlm_relative_path:
                mlm_path = os.path.join(MLM_ROOT_PATH, mlm_relative_path)
                print(f"Loading model for augmentation from {mlm_path}")
                augmenter = augmentation_config["class"](model_name_or_path=mlm_path, **augmentation_config["augmenter_parameters"])
            else:
                augmenter = augmentation_config["class"](**augmentation_config["augmenter_parameters"])
            for train_size in config["train_sizes"]:
                train_dataset, val_dataset, test_dataset = get_datasets(
                    config["dataset_name"],
                    augmenter,
                    train_size,
                    val_size=config["val_size"],
                    test_size=config["test_size"],
                    augmentation_prob=augmentation_config["augmentation_prob"],
                    load_test=config["load_test"],
                    text_columns=config["text_colname"],
                    sep_token=Tokenizer.sep_token
                )

                tokenizer = Tokenizer.from_dataset(train_dataset, num_words=10_000)
                print(f"Created tokenizer with {len(tokenizer)} vocab words. Using sequence length of {tokenizer.max_length}")
                train_dataset = DatasetWithTokenization(train_dataset, tokenizer)  # augmentation is on pure dataset
                val_dataset = DatasetWithTokenization(val_dataset, tokenizer)
                test_dataset = DatasetWithTokenization(test_dataset, tokenizer)

                data_collator = TokenizedDataCollator(label_colname=config["label_colname"],
                                                      pad_token_id=tokenizer.pad_token_id, padding=tokenizer.max_length)
                model_config = CnnClassifierConfig(num_labels=config["num_labels"], seq_len=tokenizer.max_length,
                                                   num_words=len(tokenizer), **cnn_classifier_config)
                model = CnnClassifier(model_config)

                print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
                print(train_dataset[0])
                print(val_dataset[0])
                print(test_dataset[0])

                output_dir = os.path.join(ROOT_OUTPUT_DIR, "cnn_model.pt")
                trainer_config = TrainerConfig(ckpt_path=output_dir, **trainer_config_dict)
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
                accuracies[name].append(test_result['accuracy'])
                # TODO: save csv instead of json
                with open(os.path.join(SAVE_DIR, f'{name}_{augmentation_config["name"]}_train_size_{train_size}.json'), 'w') as f:
                    json.dump(test_result, f, indent=4)
                print(test_result)
                with open(os.path.join(SAVE_DIR, 'accuracies.json'), 'w') as f:
                    json.dump(accuracies, f, indent=4)
                print(accuracies)
                shutil.rmtree(output_dir, ignore_errors=True)
                print()


if __name__ == "__main__":
    run_exp()
