import heapq
import re
from collections import Counter
from typing import List

import numpy as np
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm


class Tokenizer:
    sep_token = "[SEP]"
    unk_token = "[UNK]"
    pad_token = "[PAD]"
    special_tokens = [sep_token, unk_token, pad_token]

    def __init__(self, vocab_list: List[str], max_length: int = None):
        # add special tokens to the vocabulary
        for special_token in self.special_tokens:
            vocab_list.insert(0, special_token)
        self.word2idx = {word: idx for idx, word in enumerate(vocab_list)}
        self.idx2word = vocab_list
        # max_length used later for CNN classifier (a quantile of tokenized length distribution)
        self.max_length = max_length

    @classmethod
    def from_dataset(cls, dataset, text_column="text", num_words: int = 10_000):
        texts = [row[text_column] for row in dataset]
        tokenized_texts = [cls.tokenize(text) for text in tqdm(texts, desc="tokenization of dataset sentences")]
        vocab_list = cls.get_vocabulary(tokenized_texts, num_words)
        max_length = cls.get_max_length(tokenized_texts)
        return cls(vocab_list, max_length)

    @classmethod
    def get_vocabulary(cls, sentences: List[List[str]], num_words: int, min_occ: int = 2) -> List[str]:
        # Builds the vocabulary and keeps the "num_words" most frequent words that appeared at least min_occ times
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence)
        counter = {word: count for word, count in counter.items() if count > min_occ}
        common_words = heapq.nlargest(num_words, counter, key=counter.get)
        return common_words

    @classmethod
    def get_max_length(cls, tokenized_texts, quantile=0.995, max_max_length=512):
        lengths = [len(text) for text in tokenized_texts]
        q = int(round(np.quantile(lengths, quantile)))
        return min(q, max_max_length)

    @classmethod
    def clean_text(cls, word: str) -> str:
        """
        Cleans text: all symbols are lower-case letters. Other symbols are removed.
        If a word is a special token (like [SEP]), it is returned without modifications.
        """
        if word in cls.special_tokens:
            return word
        # Removes special symbols and keeps just lower-cased letters
        word = word.lower()
        word = re.sub(r'[^A-Za-z]+', ' ', word)
        return word

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        return word_tokenize(cls.clean_text(text))

    @property
    def pad_token_id(self):
        return self.word2idx[self.pad_token]

    @property
    def unk_token_id(self):
        return self.word2idx[self.unk_token]

    @property
    def sep_token_id(self):
        return self.word2idx[self.sep_token]

    def __len__(self):
        return len(self.idx2word)

    def __call__(self, text: str) -> List[int]:
        tokenized_text = self.tokenize(text)
        token_ids = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in tokenized_text]
        return token_ids
