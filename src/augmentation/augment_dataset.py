from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm


def augment_dataset(dataset, augmenter, n_times, text_columns, label_column: str = "label", label_map: dict = None):
    if type(text_columns) != list:
        text_columns = [text_columns]
    augmented_rows = list()
    for idx, row in enumerate(tqdm(dataset, desc="Augmenting dataset")):
        new_row = row.copy()
        new_row["augmented"] = False
        new_row["idx"] = idx
        augmented_rows.append(new_row)
        for _ in range(1, n_times + 1):
            new_row = row.copy()
            label_string = label_map[row[label_column]] if label_column and label_map else None
            for column in text_columns:  # usually no more than two columns
                try:
                    new_row[column] = augmenter(new_row[column], label=label_string)
                except Exception as e:
                    print(f"Something went wrong when augmenting item number {idx}: {e}")
                    print(new_row)
            new_row["augmented"] = True
            new_row["idx"] = idx
            augmented_rows.append(new_row)
    dataset_dict = defaultdict(list)
    columns = augmented_rows[0].keys()
    for row in augmented_rows:
        for col in columns:
            dataset_dict[col].append(row[col])
    return pd.DataFrame(dataset_dict)
