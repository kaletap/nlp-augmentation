import pandas as pd
from tqdm import tqdm

from src.augmentation import augmentations


def get_augmentation_fn(augmenter, x_cols):
    return augmenter(x_cols)


def augment_data(df, task, aug_type, aug_repeat, x_cols):
    if aug_type != "no_aug":
        augmenter = augmentations.get_augmentation_fn(aug_type, wrapped=False)
        aug_df = pd.concat([df] * (aug_repeat - 1))
        augment_text_df(aug_df, task, x_cols, augmenter)
        df = pd.concat([df, aug_df])
    else:
        df = pd.concat([df] * aug_repeat)
    return df


def augment_qa(df, idx, row, col, aug_fn):
    assert len(row["answers"]["answer_start"]) == 1, "There is more than one answer to one" \
                                                     "context. It's not supported by this preprocessing"
    answer = row["answers"]
    answer_start, answer_txt = answer["answer_start"][0], answer["text"][0]
    answer_end = answer_start + len(answer_txt)

    context_first_part = row[col][:answer_start].strip()
    context_second_part = row[col][answer_end:].strip()

    aug_context_first_part = aug_fn(context_first_part) + " "
    aug_context_second_part = " " + aug_fn(context_second_part)

    answer_start = len(aug_context_first_part)
    auged_txt = aug_context_first_part + answer_txt + aug_context_second_part

    answer["answer_start"][0] = answer_start
    df.loc[idx, col] = auged_txt
    # return df


def augment_summ(df, idx, row, col, aug_fn):
    aug_text = aug_fn(row[col])
    df.loc[idx, col] = aug_text
    # return df


def augment_text_df(df, task, x_cols, aug_fn):
    for idx, row in tqdm(df.iterrows()):
        for col in x_cols:
            if task == "qa":
                augment_qa(df, idx, row, col, aug_fn)
            elif task == "summarization":
                augment_summ(df, idx, row, col, aug_fn)
            else:
                raise ValueError(f"Task type {task} is not supported")
    return df
