import os
import shutil
import warnings
from datetime import datetime

import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

from src.data_processing import DataCollator
from src.pipelines.classification_configs import (
    augmentation_configs,
    dataset_configs,
    root_mlm_dir,
    root_output_dir,
    root_training_csv_path
)
from src.pipelines.datasets import get_datasets


# Setup
PLATFORM = "kaggle"
ROOT_OUTPUT_DIR = root_output_dir[PLATFORM]
ROOT_MLM_DIR = root_mlm_dir[PLATFORM]
ROOT_TRAINING_CSV_PATH = root_training_csv_path[PLATFORM]
SAVE_DIR = "."
# USE_FINETUNED_MODEL_FOR_CLASSIFICATION = True
TRANSFORMER_MODEL_NAME = "roberta-base"  # "albert-base-v2"
N_EXPERIMENTS = 3


# TODO: reorganize this code into neat functions
def run_exp():
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME, use_fast=False)

    results_df = pd.DataFrame(columns=["dataset", "augmentation", "training_size", "accuracy", "finetuned_clf"])
    for _ in range(N_EXPERIMENTS):
        for name, config in dataset_configs.items():
            print("Dataset:", name, "config:", config)
            mlm_relative_path = config.get("mlm_relative_path", None)
            for augmentation_config in augmentation_configs:
                print("Augmentation config:", augmentation_config)
                augmentation_name = augmentation_config["name"]
                if ROOT_TRAINING_CSV_PATH:  # we won't use augmenter since we are loading augmented data
                    augmenter = None
                else:
                    use_finetuned = augmentation_config.get("use_finetuned", None)
                    if use_finetuned and not mlm_relative_path:
                        warnings.warn(f"You are asking to use finetuned model for dataset {name} but do not provide path to the"
                                      f"pretrained model")
                    if use_finetuned and mlm_relative_path:
                        mlm_path = os.path.join(ROOT_MLM_DIR, mlm_relative_path)
                        print(f"Loading model for augmentation from {mlm_path}")
                        augmenter = augmentation_config["class"](model_name_or_path=mlm_path, **augmentation_config["augmenter_parameters"])
                    else:
                        augmenter = augmentation_config["class"](**augmentation_config["augmenter_parameters"])
                for USE_FINETUNED_MODEL_FOR_CLASSIFICATION in (False, True):
                    data_collator = DataCollator(tokenizer, text_colname="text", label_colname=config["label_colname"])
                    for train_size in config["train_sizes"]:
                        if USE_FINETUNED_MODEL_FOR_CLASSIFICATION:  # we want to load a new model for each train_size!
                            model_name_or_path = os.path.join(ROOT_MLM_DIR, mlm_relative_path)
                            print(f"Loading model for classification from {model_name_or_path}")
                        else:
                            model_name_or_path = TRANSFORMER_MODEL_NAME
                        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True,
                                                                                   num_labels=config["num_labels"])
                        training_csv_path = os.path.join(ROOT_TRAINING_CSV_PATH, name, augmentation_name, f"{train_size}.csv")
                        print("Augmenter", augmenter)
                        train_dataset, val_dataset, test_dataset = get_datasets(
                            config["dataset_name"],
                            augmenter=augmenter,
                            train_size=train_size,
                            val_size=config["val_size"],
                            test_size=config["test_size"],
                            augmentation_prob=augmentation_config["augmentation_prob"],
                            load_test=config["load_test"],
                            text_columns=config["text_colname"],
                            sep_token=tokenizer.sep_token,
                            training_csv_path=training_csv_path
                        )
                        print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
                        print(train_dataset[0])
                        print(val_dataset[0])
                        print(test_dataset[0])

                        if ROOT_TRAINING_CSV_PATH:  # less epochs since we have a lot of data
                            num_train_epochs = 2
                        else:
                            num_train_epochs = {
                                20: 10,
                                100: 10,
                                1000: 7,
                                2_500: 6,
                                10_000: 6,
                                100_000: 3
                            }.get(train_size, 6)
                            if train_size > 50_000:
                                num_train_epochs = 3

                        output_dir = os.path.join(ROOT_OUTPUT_DIR, f'{name}_{augmentation_config["name"]}_train_size_{train_size}')
                        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
                        training_args = TrainingArguments(
                            output_dir=output_dir,
                            logging_dir=SAVE_DIR,
                            num_train_epochs=num_train_epochs,
                            learning_rate=4e-5,
                            weight_decay=0.01,
                            per_device_train_batch_size=config["batch_size"],
                            per_device_eval_batch_size=config["eval_batch_size"],
                            gradient_accumulation_steps=config["gradient_accumulation_steps"],
                            warmup_steps=0,
                            logging_steps=50,
                            load_best_model_at_end=True,
                            evaluation_strategy='epoch',
                            metric_for_best_model="eval_accuracy",
                            remove_unused_columns=False,
                            no_cuda=False,
                            dataloader_num_workers=0
                        )

                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            data_collator=data_collator,
                            compute_metrics=config["metrics_function"]
                        )

                        trainer.train()

                        test_result = trainer.evaluate(test_dataset)
                        print(test_result)
                        results_df = results_df.append({
                            "dataset": name,
                            "augmentation": augmentation_config["name"],
                            "training_size": train_size,
                            "accuracy": test_result["eval_accuracy"],
                            "finetuned_clf": USE_FINETUNED_MODEL_FOR_CLASSIFICATION
                        }, ignore_index=True)
                        results_df.to_csv(os.path.join(ROOT_OUTPUT_DIR, 'results.csv'), index=False)
                        print(results_df)
                        # cleanup
                        shutil.rmtree(output_dir, ignore_errors=True)
                        print()


if __name__ == "__main__":
    run_exp()
