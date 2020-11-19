import pandas as pd
from tqdm import tqdm

from src.augmentation import augmentations


def get_augmentation_fn(augmenter, x_cols):
    return augmenter(x_cols)


def augment_data(df, aug_type, aug_repeat, x_cols):
    if aug_type != "no_aug":
        augmenter = augmentations.get_augmentation_fn(aug_type, wrapped=False)
        aug_df = pd.concat([df] * (aug_repeat - 1))
        aug_df = augment_text_df(aug_df, x_cols, augmenter)
        df = pd.concat([df, aug_df])
    else:
        df = pd.concat([df] * aug_repeat)
    return df


def augment_text_df(df, x_cols, aug_fn):
    import pdb;pdb.set_trace()
    for idx, row in tqdm(df.iterrows()):
        for col in x_cols:
            auged_text = aug_fn(row[col])
            df.loc[idx, col] = auged_text
    return df
