from functools import partial

from fastai import optimizer
import transformers
from blurr.modeling import summarization as model_sum
from blurr.modeling import question_answering as model_qa

from src.augmentation import *
from src.pipelines.metrics import compute_binary_metrics, compute_multiclass_metrics
from src.pipelines import pipeline

gigaword_config = {
    "ds_name": ("gigaword",),
    "max_len": (256, 130),
    "x_col": "document",
    "y_col": "summary",
}

cnn_dailymail_config = {
    "ds_name": ("cnn_dailymail", '3.0.0'),
    "max_len": (256, 130),
    "x_col": "article",
    "y_col": "highlights",
}

squad_v2_config = {
    "ds_name": ("squad_v2",),
    "max_len": 156,
    "x_col": ("question", "context"),
    "y_col": ("tok_answer_start", "tok_answer_end"),
}

summary_bart_config = {
    "pretrained_model_name": "facebook/bart-large-cnn", # to sie wypierdala jako czesc sciezki
    "model_class": transformers.BartForConditionalGeneration,
    "task": "summarization",
    "opt_func": optimizer.ranger,
    "loss_func": model_sum.HF_MaskedLMLoss,
    "metrics": (),
    "bs": 4,
    "pre_config_overwrite": {'max_length': 130, 'min_length': 30},
    "train_params": {
        "all": {
            "epochs": (3,),
            "unfreeze": (),
            "lr": (),
        },
    }
}

qa_bert_config = {
    "pretrained_model_name": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "model_class": transformers.BertForQuestionAnswering,
    "task": "qa",
    "opt_func": partial(optimizer.Adam, decouple_wd=True),
    "loss_func": model_qa.MultiTargetLoss,
    "metrics": (),
    "bs": 8,
    "pre_config_overwrite": {},
    "train_params": {
        "all": {
            "epochs": (6,),
            "unfreeze": (),
            "lr": (),
        },
    }
}

common_config = {
    "load_template_path": "{dataset_name}_{split}_{sample_count}_aug-{aug_type}_repeat-{repeat}.csv",
    "save_predictions": False,
    "save_model": False,
    "model_save_paths":
        "model_checkpoint_{task}_ds-{dataset}_model-{model}_train_size-{train_samples}_aug-{aug}_repeat-{repeat}_seed-{seed}",
    "metrics_save_paths":
        "metrics_{task}_ds-{dataset}_model-{model}_train_size-{train_samples}_aug-{aug}_repeat-{repeat}_seed-{seed}.csv",
    "predictions_save_paths":
        "predictions_{task}_ds-{dataset}_model-{model}_train_size-{train_samples}_aug-{aug}_repeat-{repeat}_seed-{seed}.csv",
    "targets_save_paths":
        "targets_{task}_ds-{dataset}_model-{model}_train_size-{train_samples}_aug-{aug}_repeat-{repeat}_seed-{seed}.csv",
}

experiments_setup = {
    "train_samples": ((100, 100), (1000, 10), (5000, 2)), #["all", 10, 100, 1000, 10000], # (org_smpl_count, aug_repeat)
    "augmentations": ("rules", "no_aug", ),# "vae", "rules", "style_transfer"],
    "seeds": (9, ),# 9, 11, 21, 37]
    "tasks": {
        "summarization": ((pipeline.SummarizationPipeline, {**summary_bart_config, **cnn_dailymail_config, **common_config})),
        "qa": ((pipeline.QuestionAnsweringPipeline, {**qa_bert_config, **squad_v2_config, **common_config}))
    },
}

# Classification datasets configs

ag_news_config = {
    "dataset_name": "ag_news",
    "mlm_relative_path": "ag-news-roberta/checkpoint-1400",
    "num_labels": 4,
    "text_colname": "text",
    "label_colname": "label",
    "val_size": 5_000,
    "test_size": None,
    "load_test": True,
    "batch_size": 16,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "metrics_function": compute_multiclass_metrics,
    "train_sizes": [2500]
}

imdb_config = {
    "dataset_name": "imdb",
    "mlm_relative_path": "imdb-roberta/checkpoint-300",
    "num_labels": 2,
    "text_colname": "text",
    "label_colname": "label",
    "val_size": 1_000,
    "test_size": 5_000,
    "load_test": True,
    "batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "metrics_function": compute_binary_metrics,
    "train_sizes": [2500]
}

snli_config = {
    "dataset_name": "snli",
    "mlm_relative_path": "snli-roberta/checkpoint-6400",
    "num_labels": 3,
    "text_colname": ["premise", "hypothesis"],
    "label_colname": "label",
    "val_size": 7_000,
    "test_size": None,
    "load_test": True,
    "batch_size": 16,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "metrics_function": compute_multiclass_metrics,
    "train_sizes": [2500]
}

twitter_config = {
    "dataset_name": "sentiment140",
    "mlm_relative_path": "twitter-roberta/checkpoint-2600",
    "num_labels": 2,
    "text_colname": "text",
    "label_colname": "sentiment",
    "val_size": 5_000,
    "test_size": 20_000,
    "load_test": False,
    "batch_size": 16,
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "metrics_function": compute_binary_metrics,
    "train_sizes": [2_500]
}

yelp_config = {
    "dataset_name": "yelp_polarity",
    "mlm_relative_path": "yelp-roberta/checkpoint-350",
    "num_labels": 2,
    "text_colname": "text",
    "label_colname": "label",
    "val_size": 1_000,
    "test_size": 5_000,
    "load_test": True,
    "batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "metrics_function": compute_binary_metrics,
    "train_sizes": [2_500]
}

dataset_configs = {
    "ag_news": ag_news_config,
    "imdb": imdb_config,
    "snli": snli_config,
    "twitter": twitter_config,
    "yelp": yelp_config
}

keys = ag_news_config.keys()
for config in dataset_configs.values():
    assert config.keys() == keys

# Augmentation configs (some Augmenter arguments depend on the dataset, for example pretrained model)
# That's why some of them have to be specified in the code

mlm_insertion_config = {
    "name": "mlm_insertion",
    "class": MLMInsertionAugmenter,
    "use_finetuned": True,
    "augmenter_parameters": {
        "fraction": 0.12,
        "min_mask": 1,
        "max_mask": 100,
        "topk": 10,
        "uniform": False
    },
    "augmentation_prob": 0.7,
}

mlm_substitution_config = {
    "name": "mlm_substitution",
    "class": MLMSubstitutionAugmenter,
    "use_finetuned": True,
    "augmenter_parameters": {
        "fraction": 0.12,
        "min_mask": 1,
        "max_mask": 100,
        "topk": 10,
        "uniform": False
    },
    "augmentation_prob": 0.7,
}

swap_config = {
    "name": "random_swap",
    "class": RandomWordAugmenter,
    "augmenter_parameters": {
        "action": "swap",
        "aug_p": 0.2,
        "aug_min": 1,
        "aug_max": 10,
    },
    "augmentation_prob": 0.7
}

bart_augmenter_config = {
    "name": "bart_augmentation",
    "class": BartAugmenter,
    "augmenter_parameters": {
        "fraction": 0.2,
        "min_mask": 1,
        "lambda_": 2.5,
        "num_beams": 1
    },
    "augmentation_prob": 0.7,
}

no_augmenter_config = {
    "name": "no_augmentation",
    "class": NoAugmenter,
    "augmenter_parameters": dict(),
    "augmentation_prob": 0.0,
}
