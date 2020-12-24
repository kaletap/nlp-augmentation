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
    "max_len": 490,
    "x_col": ("question", "context"),
    "y_col": ("tok_answer_start", "tok_answer_end"),
}

covid_squad_config = {
    "ds_name": ("covid_squad",),
    "max_len": 490,
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
    "bs": 4,
    "pre_config_overwrite": {'max_length': 130, 'min_length': 30},
    "train_params": {
        "all": {
            "epochs": (1,),
            "unfreeze": (),
            "lr": (),
        },
    }
}

qa_xlm_config = {
    "pretrained_model_name": "xlm-mlm-ende-1024",
    "model_class": transformers.XLMForQuestionAnswering,
    "task": "qa",
    "opt_func": partial(optimizer.Adam, decouple_wd=True),
    "loss_func": model_qa.MultiTargetLoss,
    "metrics": (),
    "bs": 8,
    "pre_config_overwrite": {},
    "train_params": {
        "all": {
            "epochs": (10,),
            "unfreeze": (),
            "lr": (),
        },
    }
}

qa_distilbert_config = {
    "pretrained_model_name": "distilbert-base-uncased",
    "model_class": transformers.DistilBertForQuestionAnswering,
    "task": "qa",
    "opt_func": partial(optimizer.Adam, decouple_wd=True),
    "loss_func": model_qa.MultiTargetLoss,
    "metrics": (),
    "bs": 8,
    "pre_config_overwrite": {},
    "train_params": {
        "all": {
            "epochs": (10,),
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
    "bs": 4,
    "pre_config_overwrite": {},
    "train_params": {
        "all": {
            "epochs": (5,),
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
    "train_samples": ((500, 500), (500, 1000), (500, 2500), (500, 5000), (500, 10000)), # , (100, 100), (1500, 7),  #  # (org_smpl_count, aug_repeat)
    "augmentations": ("LM", ),# "vae", "rules", "style_transfer"], # "no_aug", "rules",
    "seeds": (11, 21),# 9, 11, 21, 37]
    "tasks": {
        "summarization": ((pipeline.SummarizationPipeline, {**summary_bart_config, **cnn_dailymail_config, **common_config})),
        "qa": ((pipeline.QuestionAnsweringPipeline, {**qa_bert_config, **squad_v2_config, **common_config})),
        "qa_covid": ((pipeline.QuestionAnsweringPipeline, {**qa_bert_config, **covid_squad_config, **common_config}))
    },
}
