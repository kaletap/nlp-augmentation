from functools import partial

from fastai import optimizer
import transformers
from blurr.modeling import summarization as model_sum
from blurr.modeling import question_answering as model_qa

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
    "pretrained_model_name": "facebook/bart-large-cnn",
    "model_class": transformers.BartForConditionalGeneration,
    "task": "summarization",
    "opt_func": optimizer.ranger,
    "loss_func": model_sum.HF_MaskedLMLoss,
    "metrics": (),
    "bs": 2,
    "pre_config_overwrite": {'max_length': 130, 'min_length': 30},
    "train_params": {
        "all": {
            "epochs": (1,),
            "unfreeze": (),
            "lr": (),
        },
        1000: {
            "epochs": (1, 1),
            "unfreeze": (-1,),
            "lr": ((10, 1),),
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
            "epochs": (1,),
            "unfreeze": (),
            "lr": (),
        },
        1000: {
            "epochs": (1,),
            "unfreeze": (),
            "lr": (),
        },
    }
}

common_config = {
    "save_predictions": False,
    "model_save_paths":
        "model_checkpoint_{task}_ds-{dataset}_model-{model}_train_size-{train_samples}_aug-{aug}_seed-{seed}.pkl",
    "metrics_save_paths":
        "metrics_{task}_ds-{dataset}_model-{model}_train_size-{train_samples}_aug-{aug}_seed-{seed}.csv",
    "predictions_save_paths":
        "predictions_{task}_ds-{dataset}_model-{model}_train_size-{train_samples}_aug-{aug}_seed-{seed}.csv",
    "targets_save_paths":
        "targets_{task}_ds-{dataset}_model-{model}_train_size-{train_samples}_aug-{aug}_seed-{seed}.csv",
}

experiments_setup = {
    "train_samples": (1000,), #["all", 10, 100, 1000, 10000],
    "augmentations": ("no_aug",),# "vae", "rules", "style_transfer"],
    "seeds": (1990,),# 9, 11, 21, 37]
    "tasks": {
        "summarization": ((pipeline.SummarizationPipeline, {**summary_bart_config, **cnn_dailymail_config, **common_config})),
        "qa": ((pipeline.QuestionAnsweringPipeline, {**qa_bert_config, **squad_v2_config, **common_config}))
    }
}

# Classification datasets configs

ag_news_config = {
    "dataset_name": "ag_news",
    "num_labels": 4,
    "text_colname": "text",
    "label_colname": "label",
    "val_size": 5_000,
    "test_size": None,
    "load_test": True,
    "batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "metrics_function": compute_multiclass_metrics,
    "train_sizes": []
}

imdb_config = {
    "dataset_name": "imdb",
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
    "train_sizes": []
}

snli_config = {
    "dataset_name": "snli",
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
    "train_sizes": [20, 100, 1_000, 2_500]
}

twitter_config = {
    "dataset_name": "sentiment140",
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
    "train_sizes": [20, 100, 1_000, 2_500]
}

yelp_config = {
    "dataset_name": "yelp_polarity",
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
    "train_sizes": [20, 100, 1_000]
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
