from src.augmentation import *
from src.pipelines.metrics import compute_binary_metrics, compute_multiclass_metrics

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
    "train_sizes": [20, 100, 1000, 2_500],
    "label_map": {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech"
    }
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
    "train_sizes": [20, 100, 1000],
    "label_map": {
        0: "negative",
        1: "positive"
    }
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
    "batch_size": 20,
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "metrics_function": compute_multiclass_metrics,
    "train_sizes": [20, 100, 1000, 2_500],
    "label_map": {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }
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
    "train_sizes": [20, 100, 1000, 2_500],
    "label_map": {
        0: "negative",
        1: "positive"
    }
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
    "train_sizes": [20, 100, 1000, 2500],
    "label_map": {
        0: "negative",
        1: "positive"
    }
}

dataset_configs = {
    "ag_news": ag_news_config,
    # "imdb": imdb_config,
    "snli": snli_config,
    # "twitter": twitter_config,
    # "yelp": yelp_config
}

keys = ag_news_config.keys()
for config in dataset_configs.values():
    assert config.keys() == keys, "Dataset configs have to have the same keys"

# Augmentation configs (some Augmenter arguments depend on the dataset, for example pretrained model)
# That's why some of them have to be specified in the code

mlm_insertion_config = {
    "name": "mlm_insertion",
    "class": MLMInsertionAugmenter,
    "use_finetuned": False,
    "augmenter_parameters": {
        "min_mask": 1,
        "max_mask": 100,
        "uniform": False
    },
    "augmentation_prob": 0.7,
}

mlm_substitution_config = {
    "name": "mlm_substitution",
    "class": MLMSubstitutionAugmenter,
    "use_finetuned": False,
    "augmenter_parameters": {
        "min_mask": 1,
        "max_mask": 100,
        "uniform": False
    },
    "augmentation_prob": 0.7,
}

finetuned_mlm_insertion_config = {
    "name": "finetuned_mlm_insertion",
    "class": MLMInsertionAugmenter,
    "use_finetuned": True,
    "augmenter_parameters": {
        "min_mask": 1,
        "max_mask": 100,
        "uniform": False
    },
    "augmentation_prob": 0.7,
}

finetuned_mlm_substitution_config = {
    "name": "finetuend_mlm_substitution",
    "class": MLMSubstitutionAugmenter,
    "use_finetuned": True,
    "augmenter_parameters": {
        "min_mask": 1,
        "max_mask": 100,
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

wordnet_config = {
    "name": "wordnet",
    "class": RuleBasedAugmenter,
    "augmenter_parameters": dict(),
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

# Classification model config

augmentation_configs = [no_augmenter_config, swap_config, wordnet_config, mlm_insertion_config, mlm_substitution_config,
                        finetuned_mlm_insertion_config, finetuned_mlm_insertion_config]

cnn_classifier_config = {
    "embedding_size": 128,
    "out_size": 256,
    "stride": 2
}

# Classification trainer config

trainer_config_dict = {
    "batch_size": 64,
    "cuda": True,
    "learning_rate": 0.001,
    "weight_decay": 0.01
}

# Directories

root_output_dir = {
    "kaggle": "/kaggle/working",
    "colab": "/content/drive/MyDrive/Colab Notebooks/nlp/results",
    "local": "examples"
}

root_mlm_dir = {
    "kaggle": "/kaggle/input/",
    "colab": "/content/drive/MyDrive/Colab Notebooks/nlp/pretrained_models",
    "local": "../mgr-code"
}

root_training_csv_path = {
    "kaggle": "/kaggle/input/classification/",
    "colab": "/content/drive/MyDrive/Colab Notebooks/nlp/data",
    "local": "examples"
}
