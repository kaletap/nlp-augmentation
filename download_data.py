import datasets
import pandas as pd

from src.training_utils import data_downloading

PARAMS_LIST = [
    [
        "squad_v2",
        "train",
        [
             # (100, ("context",), {"type": "no_aug", "repeat": 50}),
             # (1000, ("context",), {"type": "no_aug", "repeat": 5}),
             # (100, ("context",), {"type": "rules", "repeat": 50}),
             # (1000, ("context",), {"type": "rules", "repeat": 5}),
             (100, ("context",), {"type": "LM", "repeat": 50}),
             (1000, ("context",), {"type": "LM", "repeat": 5})
        ],
    ],
    [
        "squad_v2",
        "validation",
        [
            # ("all", ("context",), {"type": "no_aug", "repeat": 1}),
            # ("all", ("context",), {"type": "no_aug", "repeat": 1})
        ],
    ]
]


for (dataset_name, split, ds_params) in PARAMS_LIST:
    ds = datasets.load_dataset(dataset_name, split=split)
    df = pd.DataFrame(ds)
    for sample_count, x_cols, aug in ds_params:
        print(f"start processing {aug['type']}, sample_count {sample_count}")
        if isinstance(sample_count, int):
            df_subset = df[:sample_count] # make sure its a copy and does not modify original dataset
            print(sample_count)
        else:
            df_subset = df
            print("not trimming ds cuz sample_count: " + sample_count)
        auged_df = data_downloading.augment_data(
            df_subset, aug_type=aug["type"], aug_repeat=aug["repeat"], x_cols=x_cols)
        auged_df.to_csv(f"{dataset_name}_{split}_{sample_count}_aug-{aug['type']}_repeat-{aug['repeat']}.csv")
