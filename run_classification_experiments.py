import json
import os
from collections import defaultdict

# %pip install datasets
# %pip install transformers
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

from src.augmentation import MLMInsertionAugmenter, MLMSubstitutionAugmenter
from src.data_processing import DataCollator
from src.pipelines.configs import dataset_configs
from src.pipelines.datasets import get_datasets


# Setup
output_dir = '/kaggle/temp/'
save_dir = "."
AUGMENTATION = "mlm_substitution"
AUGMENTATION_PROB = 0.7
augmentation_config = {
    "fraction": 0.12,
    "min_mask": 1,
    "max_mask": 100,
    "topk": 10,
    "uniform": False
}


def run_exp():
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=False)
    # we can also use task fine-tuned language model if we want
    mlm_model = AutoModelForMaskedLM.from_pretrained('roberta-base', return_dict=True).eval()
    augmenter = MLMSubstitutionAugmenter(mlm_model, tokenizer, **augmentation_config)

    accuracies = defaultdict(list)
    print(augmentation_config)
    for name, config in dataset_configs.items():
        print(name, config)
        for train_size in config["train_sizes"]:
            data_collator = DataCollator(tokenizer, text_colname=config["text_colname"], label_colname=config["label_colname"])
            model = AutoModelForSequenceClassification.from_pretrained('roberta-base', return_dict=True, num_labels=config["num_labels"])

            train_dataset, val_dataset, test_dataset = get_datasets(
                config["dataset_name"],
                augmenter,
                train_size,
                val_size=config["val_size"],
                test_size=config["test_size"],
                augmentation_prob=AUGMENTATION_PROB,
                load_test=config["load_test"]
            )
            print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
            print(train_dataset[0])
            print(val_dataset[0])
            print(test_dataset[0])

            num_train_epochs = {
                20: 10,
                100: 10,
                1000: 8,
                10_000: 6,
                100_000: 3
            }.get(train_size, 6)

            # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                logging_dir=save_dir,
                num_train_epochs=num_train_epochs,
                learning_rate=3e-5,
                weight_decay=0.01,
                per_device_train_batch_size=config["batch_size"],
                per_device_eval_batch_size=config["batch_size"],
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                warmup_steps=0,
                logging_steps=10,
                load_best_model_at_end=True,
                evaluation_strategy='epoch',
                metric_for_best_model="eval_accuracy",
                remove_unused_columns=False,
                no_cuda=False,
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
            with open(os.path.join(save_dir, f'{name}_{AUGMENTATION}_train_size_{train_size}.json'), 'w') as f:
                json.dump(test_result, f)
            print(test_result)
            with open(os.path.join(save_dir, 'accuracies.json'), 'w') as f:
                json.dump(test_result, f)
            print()


if __name__ == "__main__":
    run_exp()
