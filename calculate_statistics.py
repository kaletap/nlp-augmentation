import argparse
import os
from copy import deepcopy
import numpy as np
import pandas as pd
import collections

from src.pipelines import configs
from src.training_utils import training_utils

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="qa_covid",
                    choices=list(configs.experiments_setup["tasks"].keys()), help="type of task")
parser.add_argument("--data_path", type=str, default="./data/covid_squad_results/aug_method_comparision",
                    help="path to folder with data")
parser.add_argument("--save_path", type=str, default="qa_results.csv",
                    help="path to save results")
args = parser.parse_args()


experiments_setup = {
    "ds_name": ("covid_squad",),
    "task": {"qa_covid": ("valid_loss", "exact_match", "f1"), "summarization": ("valid_loss", "rouge_1", "rouge_2", "rouge_L")},
    "train_samples": ((100, 100), (500, 20), (1500, 7)),
    "augmentations": ("rules", "no_aug", "LM"),
    "seeds": (9, ),
    "pretrained_model_name": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "metrics_save_paths":
        "metrics_{task}_ds-{dataset}_model-{model}_train_size-{train_samples}_aug-{aug}_repeat-{repeat}_seed-{seed}.csv",
}


def calculate_agg_metrics(exp_values, agg_cols):
    results = {}
    for col in agg_cols:
        results[col + "_mean"] = np.array(exp_values[col]).mean()
        results[col + "_std"] = np.array(exp_values[col]).std()
    return results


def aggregate_exp_result_by_seed(data_path, row_csv_paths, agg_cols):
    exp_values = collections.defaultdict(list)
    for csv_path in row_csv_paths:
        df = pd.read_csv(os.path.join(data_path, csv_path))
        row = df[df.valid_loss == df.valid_loss.max()]
        for col in agg_cols:
            exp_values[col].append(row[col])
    return exp_values


def prepare_paths(task, config):
    row_csv_paths = []
    aug = config["augmentation"]
    train_samples = config["train_samples"]
    for seed in config["seeds"]:
        config["seed"] = seed
        config_cp = deepcopy(config)
        config_cp = training_utils.fill_paths(task, train_samples, aug, seed, config_cp, paths=("metrics_save_paths",))
        row_csv_paths.append(config_cp["metrics_save_paths"])
    return row_csv_paths


def prepare_df_row(task, config, data_path):
    agg_cols = config["task"][task]
    row_csv_paths = prepare_paths(task, config)
    exp_values = aggregate_exp_result_by_seed(
        data_path=data_path, row_csv_paths=row_csv_paths, agg_cols=agg_cols
    )
    return calculate_agg_metrics(exp_values, agg_cols)


def summraize_results(task, data_path, save_path, og_config):
    row_list = []
    config = deepcopy(og_config)
    for aug in og_config["augmentations"]:
        config["augmentation"] = aug
        for train_samples, reps in og_config["train_samples"]:
            config["train_samples"] = train_samples
            config["repeat"] = reps

            results = prepare_df_row(task, config, data_path)
            results["augmentation"] = aug
            results["train_samples"] = train_samples
            results["task"] = task
            row_list.append(results)

    df = pd.DataFrame(row_list).round(3)
    df.to_csv(save_path)


summraize_results(task=args.task, data_path=args.data_path, save_path=args.save_path, og_config=experiments_setup)
