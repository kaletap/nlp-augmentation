import datasets
import pandas as pd

from src.training_utils import data_downloading

FOLDER = "/kaggle/input/covid-squad"

max_len = 300

PARAMS_LIST = [
    # [
    #     ("cnn_dailymail", '3.0.0'),
    #     "train",
    #     [
    #         # (100, "summarization", ("article",), {"type": "no_aug", "repeat": 100}),
    #         # (1000, "summarization", ("article",), {"type": "no_aug", "repeat": 10}),
    #         # (5000, "summarization", ("article",), {"type": "no_aug", "repeat": 2}),
    #         # (100, "summarization", ("article",), {"type": "rules", "repeat": 100}),
    #         # (1000, "summarization", ("article",), {"type": "rules", "repeat": 10}),
    #         # (5000, "summarization", ("article",), {"type": "rules", "repeat": 2}),
    #         (100, "summarization", ("article",), {"type": "LM", "repeat": 100}),
    #         (1000, "summarization", ("article",), {"type": "LM", "repeat": 10}),
    #         (5000, "summarization", ("article",), {"type": "LM", "repeat": 2}),
    #     ],
    # ],
    # [
    #     ("cnn_dailymail", '3.0.0'),
    #     "validation",
    #     [
    #         (10000, "summarization", ("article",), {"type": "no_aug", "repeat": 1}),
    #     ],
    # ],
    # [
    #     ("squad_v2",),
    #     "train",
    #     [
    #          ("all", "qa", ("context",), {"type": "no_aug", "repeat": 1}),
    #          (100, "qa", ("context",), {"type": "no_aug", "repeat": 100}),
    #          (1000, "qa", ("context",), {"type": "no_aug", "repeat": 10}),
    #          (5000, "qa", ("context",), {"type": "no_aug", "repeat": 2}),
    #          (100, "qa", ("context",), {"type": "rules", "repeat": 100}),
    #          (1000, "qa", ("context",), {"type": "rules", "repeat": 10}),
    #          (5000, "qa", ("context",), {"type": "rules", "repeat": 10}),
    #          (100, "qa", ("context",), {"type": "LM", "repeat": 100}), # 100 original and 9900 augmented
    #          (1000, "qa", ("context",), {"type": "LM", "repeat": 10}), # 1000 original and 9000 augmented
    #          (5000, "qa", ("context",), {"type": "LM", "repeat": 10}), # 5000 original and 5000 augmented
    #     ],
    # ],
    # [
    #     ("squad_v2",),
    #     "validation",
    #     [
    #          ("all", "qa", ("context",), {"type": "no_aug", "repeat": 1}),
    #     ],
    # ]
    [
        ("covid_squad",),
        "train",
        [
             # ("all", "qa", ("context",), {"type": "no_aug", "repeat": 1}),
             # (100, "qa", ("context",), {"type": "no_aug", "repeat": 100}),
             # (500, "qa", ("context",), {"type": "no_aug", "repeat": 20}),
             # (1500, "qa", ("context",), {"type": "no_aug", "repeat": 7}),
             # (100, "qa", ("context",), {"type": "rules", "repeat": 100}),
             # (500, "qa", ("context",), {"type": "rules", "repeat": 20}),
             # (1500, "qa", ("context",), {"type": "rules", "repeat": 7}),
             (100, "qa", ("context",), {"type": "LM", "repeat": 100}), # 100 original and 9900 augmented
             # (500, "qa", ("context",), {"type": "LM", "repeat": 20}), # 1000 original and 9000 augmented
             # (1500, "qa", ("context",), {"type": "LM", "repeat": 7}), # 5000 original and 5000 augmented
        ],
    ],
    # [
    #     ("covid_squad",),
    #     "validation",
    #     [
    #          ("all", "qa", ("context",), {"type": "no_aug", "repeat": 1}),
    #     ],
    # ]
]


def load_df(dataset_name, split):
    if "covid" in dataset_name[0]:
        df = pd.read_csv(f"{FOLDER}/{split}_COVID-QA.csv")
    else:
        ds = datasets.load_dataset(*dataset_name, split=split)
        df = pd.DataFrame(ds)
    return df


for (dataset_name, split, ds_params) in PARAMS_LIST:
    df = load_df(dataset_name, split)
    for sample_count, task, x_cols, aug in ds_params:
        print(f"start processing {aug['type']}, sample_count {sample_count}")
        if dataset_name == "squad_v2":
            df["is_impossible"] = df["answers"].apply(lambda x: len(x["answer_start"]) == 0)
            df = df[df.is_impossible == False]
            df = df.drop(columns=["is_impossible"])
        if isinstance(sample_count, int):
            df_subset = df[:sample_count]
            print(sample_count)
        else:
            df_subset = df
            print("not trimming ds cuz sample_count: " + sample_count)
        auged_df = data_downloading.augment_data(
            df_subset, task=task, aug_type=aug["type"], aug_repeat=aug["repeat"], x_cols=x_cols, max_len=max_len)
        auged_df.to_csv(f"{'-'.join(dataset_name)}_{split}_{sample_count}_aug-{aug['type']}_repeat-{aug['repeat']}.csv")
