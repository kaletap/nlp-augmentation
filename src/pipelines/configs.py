from functools import partial

from fastai import optimizer
import transformers
from blurr import modeling as blurr_model

from src.pipelines import pipeline

gigaword_config = {
    "ds_name": "gigaword",
    "max_len": [256, 130],
    "x_col": "",
    "y_col": "",
}

squad_v2_config = {
    "ds_name": "squad_v2",
    "max_len": 256,
    "x_col": "",
    "y_col": "",
}

summary_bart_config = {
    "pretrained_model_name": "facebook/bart-large-cnn",
    "model_class": transformers.BartForConditionalGeneration,
    "task": "summarization",
    "opt_func": optimizer.ranger,
    "loss_func": blurr_model.summarization.HF_MaskedLMLoss(),
    "metrics": [],
    "bs": 8,
    "pre_config_overwrite": {'max_length': 130, 'min_length': 30},
    "train_params": { # should it depend on ammount of data?
        {10000: {
            "epochs": [2, 1, 1],
            "unfreeze": [-3, "all"]},
            "lr": [lambda x: x / 10, lambda x: slice(x / 1000, x / 100)],
            "moms": [(0.8, 0.7), (0.8, 0.7)], # [None, None]
        },
    }
}

qa_bert_config = {
    "pretrained_model_name": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "model_class": transformers.BertForQuestionAnswering,
    "task": "qa",
    "opt_func": partial(optimizer.Adam, decouple_wd=True),
    "loss_func": blurr_model.question_anwsering.MultiTargetLoss(),
    "metrics": [],
    "bs": 8,
    "pre_config_overwrite": {},
    "train_params": {
        {10000: {
            "epochs": [2, 1, 1],
            "unfreeze": [-3, "all"]},
            "lr": [lambda x: x/10, lambda x: slice(x/1000, x/100)],
            # "moms": [(0.8, 0.7), (0.8, 0.7)]
        },
        # {1000: {"epochs": [3, 2, 1], "unfreeze": [False, True, True], "lr": [lambda x:x, lambda x:slice(x/1000, x/100)]}},
    }
    # hmm moments?
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
    "train_samples": [False], #[10, 100, 1000, 10000],
    "augmentations": ["no_aug"],# "vae", "rules", "style_transfer"],
    "seeds": [1990],# 9, 11, 21, 37]
    "tasks": {
        "summarization": [(pipeline.QuestionAnsweringPipeline, {**summary_bart_config, **gigaword_config, **common_config})],
        "qa": [(pipeline.SummarizationPipeline, {**qa_bert_config, **squad_v2_config, **common_config})]
    }
}
