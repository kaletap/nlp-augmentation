from typing import List

import torch


class DataCollator:
    def __init__(self, tokenizer, text_colname='text', label_colname='label'):
        self.tokenizer = tokenizer
        self.text_colname = text_colname
        self.label_colname = label_colname

    def __call__(self, examples: List[dict]):
        if self.label_colname == 'sentiment':
            labels = [0 if example[self.label_colname] == 0 else 1 for example in examples]
        else:
            labels = [example[self.label_colname] for example in examples]
        texts = [example[self.text_colname] for example in examples]
        tokenizer_output = self.tokenizer(texts, truncation=True, padding=True)
        return {
            'labels': torch.tensor(labels),
            'input_ids': torch.tensor(tokenizer_output['input_ids']),
            'attention_mask': torch.tensor(tokenizer_output['attention_mask'])
        }
