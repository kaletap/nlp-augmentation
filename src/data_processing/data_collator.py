from typing import List, Tuple

import torch


class DataCollator:
    def __init__(self, tokenizer, text_colname='text', label_colname='label'):
        self.tokenizer = tokenizer
        self.text_colname = text_colname
        self.label_colname = label_colname

    def __call__(self, examples: List[dict]) -> dict:
        if self.label_colname == 'sentiment':  # a hack for sentiment140 dataset
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


class TokenizedDataCollator:
    def __init__(self, text_colname="tokenized_text", label_colname="label", pad_token_id=0, padding=None, max_length=512):
        self.text_colname = text_colname
        self.label_colname = label_colname
        self.pad_token_id = pad_token_id
        self.padding = padding
        self.max_length = max_length

    def __call__(self, examples: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.label_colname == 'sentiment':  # a hack for sentiment140 dataset
            labels = [0 if example[self.label_colname] == 0 else 1 for example in examples]
        else:
            labels = [example[self.label_colname] for example in examples]
        tokenized_texts = [example[self.text_colname] for example in examples]
        seq_length = self.padding or max(len(seq) for seq in tokenized_texts)
        seq_length = min(seq_length, self.max_length)
        padded_texts = [self.pad(seq, seq_length) for seq in tokenized_texts]
        return torch.tensor(padded_texts), torch.tensor(labels)

    @staticmethod
    def pad(seq, seq_length):
        to_pad = seq_length - len(seq)
        padded_seq = seq + [0]*to_pad
        return padded_seq[:seq_length]
