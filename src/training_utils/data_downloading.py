import re
import pandas as pd
from tqdm import tqdm

from src.augmentation import augmentations


def get_augmentation_fn(augmenter, x_cols):
    return augmenter(x_cols)


def augment_data(df, task, aug_type, aug_repeat, x_cols, max_len):
    if aug_type != "no_aug":
        augmenter = augmentations.get_augmentation_fn(aug_type, wrapped=False)
        aug_df = pd.concat([df] * (aug_repeat - 1)).reset_index(drop=True)
        augment_text_df(aug_df, task, x_cols, augmenter, max_len)
        df = pd.concat([df, aug_df]).reset_index(drop=True)
    else:
        df = pd.concat([df] * aug_repeat).reset_index(drop=True)
    return df


def augment_qa(df, idx, row, col, aug_fn, max_len):
    row["answers"] = eval(row["answers"])
    assert len(row["answers"]["answer_start"]) == 1, "There is more than one answer to one" \
                                                     "context. It's not supported by this preprocessing"
    answer = row["answers"]
    answer_start, answer_txt = answer["answer_start"][0], answer["text"][0]
    text = row[col]

    ans_word_count = len(answer_txt.split())
    max_len = (max_len - ans_word_count) // 2

    context_first_part = " ".join(text[:answer_start].strip().split()[-max_len:])
    context_second_part = " ".join(text[answer_start + len(answer_txt):].strip().split()[:max_len])

    aug_context_first_part = aug_fn(context_first_part) + " "
    aug_context_second_part = " " + aug_fn(context_second_part)
    auged_txt = aug_context_first_part + answer_txt + aug_context_second_part

    answer["answer_start"][0] = len(aug_context_first_part)
    df.loc[idx, col] = auged_txt
    return df


def augment_summ(df, idx, row, col, aug_fn):
    aug_text = aug_fn(row[col])
    df.loc[idx, col] = aug_text
    return df


def augment_text_df(df, task, x_cols, aug_fn, max_len):
    for idx, row in tqdm(df.iterrows()):
        for col in x_cols:
            if task == "qa":
                df = augment_qa(df, idx, row, col, aug_fn, max_len)
            elif task == "summarization":
                df = augment_summ(df, idx, row, col, aug_fn)
            else:
                raise ValueError(f"Task type {task} is not supported")
    return df
