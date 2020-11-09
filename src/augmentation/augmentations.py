from abc import ABC, abstractmethod
import heapq
import random

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class RuleBasedAugmenter:
    pass


class MLMAugmenter(ABC):
    def __init__(self, model=None, tokenizer=None, fraction: float = 0.12, min_mask: int = 1, max_mask: int = 100,
                 topk: int = 5, uniform: bool = False, device=None):
        """
        :param model: huggingface/transformers model for masked language modeling
            e.g model = RobertaForMaskedLM.from_pretrained('roberta-base', return_dict=True)
        :param tokenizer: huggingface/transformers tokenizer
            e.g tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        :param fraction: fraction of words to insert
        :param min_mask: minimum number of <mask> tokens to insert
        :param max_mask: maximum number ot tokens to mask
        :param topk: number of top words to sample from
        :param uniform: whether to sample uniformly from topk words (defaults to False)
        :param device: torch.device
        """
        self.device = device or torch.device('cpu')
        model = model or AutoModelForMaskedLM.from_pretrained('roberta-base', return_dict=True)
        self.model = model.eval().to(self.device)
        tokenizer = tokenizer or AutoTokenizer.from_pretrained('roberta-base', use_fast=False)
        self.tokenizer = tokenizer
        self.vocab_words = list(tokenizer.get_vocab().keys())
        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id
        self.topk = topk
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.uniform = uniform
        self.fraction = fraction

    def sample_word(self, predicted_probas, black_word=None):
        if hasattr(predicted_probas, 'tolist'):
            predicted_probas = predicted_probas.tolist()
        if black_word:
            most_probable = heapq.nlargest(self.topk + 1, zip(self.vocab_words, predicted_probas), key=lambda t: t[1])
            most_probable = [t for t in most_probable if t[0] != black_word]
        else:
            most_probable = heapq.nlargest(self.topk, zip(self.vocab_words, predicted_probas), key=lambda t: t[1])

        words, probas = zip(*most_probable)
        word = random.choice(words) if self.uniform else random.choices(words, weights=probas)[0]
        return self.tokenizer.convert_tokens_to_string(word).strip()

    @abstractmethod
    def __call__(self, text: str):
        pass


class MLMInsertionAugmenter(MLMAugmenter):
    def __call__(self, text: str):
        if self.fraction == 0:
            return text
        words = np.array(text.split(), dtype='object')
        n_mask = max(self.min_mask, int(len(words) * self.fraction))
        n_mask = min(n_mask, self.max_mask)
        max_masked_idx = min(self.tokenizer.model_max_length // 2 - n_mask,
                             len(words) + 1)  # offset, since lenght might increase after tokenization
        # end of the long text won't be augmented, but I guess we can live with that
        masked_indices = np.sort(np.random.choice(max_masked_idx, size=n_mask, replace=False))
        masked_words = np.insert(words, masked_indices, self.mask_token)
        masked_text = " ".join(masked_words)

        tokenizer_output = self.tokenizer([masked_text], truncation=True)
        input_ids = torch.tensor(tokenizer_output['input_ids']).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            predicted_logits = output.logits[input_ids == self.mask_token_id]
            predicted_probas = predicted_logits.softmax(1)

        predicted_words = [self.sample_word(probas).strip() for probas in predicted_probas]

        new_words = np.insert(words, masked_indices, predicted_words)
        new_text = " ".join(new_words)
        return new_text


class MLMSubstitutionAugmenter(MLMInsertionAugmenter):
    def __call__(self, text: str):
        if self.fraction == 0:
            return text
        words = np.array(text.split(), dtype='object')
        vocab_word_indices = [i for i, word in enumerate(words) if word in self.vocab_words]
        if not vocab_word_indices:
            return text
        n_mask = max(self.min_mask, int(len(words) * self.fraction))
        n_mask = min(n_mask, self.max_mask, len(vocab_word_indices))
        max_masked_idx = min(self.tokenizer.model_max_length // 2 , len(words) + 1)  # offset, since lenght might increase after tokenization
        # end of the long text won't be augmented, but I guess we can live with that
        masked_indices = np.sort(np.random.choice(vocab_word_indices, size=n_mask, replace=False))
        masked_words = words[masked_indices]
        words[masked_indices] = self.mask_token
        masked_text = " ".join(words)

        tokenizer_output = self.tokenizer([masked_text], truncation=True)
        input_ids = torch.tensor(tokenizer_output['input_ids']).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            predicted_logits = output.logits[input_ids == self.mask_token_id]
            predicted_probas = predicted_logits.softmax(1)
        predicted_words = [self.sample_word(probas, black_word=word).strip() for probas, word in zip(predicted_probas, masked_words)]
        words[masked_indices] = predicted_words
        new_text = " ".join(words)
        return new_text
