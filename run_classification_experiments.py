import json
import os
import shutil
from collections import defaultdict
from datetime import datetime

from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

from src.data_processing import DataCollator
from src.pipelines.configs import dataset_configs
from src.pipelines.configs import bart_augmenter_config as augmentation_config
from src.pipelines.datasets import get_datasets


# Setup
ROOT_OUTPUT_DIR = '/kaggle/temp/'
SAVE_DIR = "."


def run_exp():
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=False)
    augmenter = augmentation_config["class"](**augmentation_config["augmenter_parameters"])

    accuracies = defaultdict(list)
    print(augmentation_config)
    for name, config in dataset_configs.items():
        print(name, config)
        for train_size in config["train_sizes"]:
            data_collator = DataCollator(tokenizer, text_colname="text", label_colname=config["label_colname"])
            model = AutoModelForSequenceClassification.from_pretrained('roberta-base', return_dict=True, num_labels=config["num_labels"])

            train_dataset, val_dataset, test_dataset = get_datasets(
                config["dataset_name"],
                augmenter,
                train_size,
                val_size=config["val_size"],
                test_size=config["test_size"],
                augmentation_prob=augmentation_config["augmentation_prob"],
                load_test=config["load_test"],
                text_columns=config["text_colname"],
                sep_token=tokenizer.sep_token
            )
            print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
            print(train_dataset[0])
            print(val_dataset[0])
            print(test_dataset[0])

            num_train_epochs = {
                20: 10,
                100: 10,
                1000: 7,
                2_500: 6,
                10_000: 6,
                100_000: 3
            }.get(train_size, 6)

            output_dir = os.path.join(ROOT_OUTPUT_DIR, f'{name}_{augmentation_config["name"]}_train_size_{train_size}')
            # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                logging_dir=SAVE_DIR,
                num_train_epochs=num_train_epochs,
                learning_rate=3e-5,
                weight_decay=0.01,
                per_device_train_batch_size=config["batch_size"],
                per_device_eval_batch_size=config["eval_batch_size"],
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                warmup_steps=0,
                logging_steps=100,
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
            accuracies[name].append(test_result['eval_accuracy'])
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
