from abc import ABC, abstractmethod
from functools import partial
import heapq
import itertools
import random

import nlpaug.augmenter.word as naw
import numpy as np
import torch
from fastai.data import transforms
from transformers import AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer


def get_augmentation_fn(aug_name, wrapped=True):
    if aug_name == "no_aug":
        return transforms.ColReader
    elif aug_name == "rules":
        augmenter = RuleBasedAugmenter()
    elif aug_name == "LM":
        augmenter = MLMSubstitutionAugmenter()
    else:
        raise ValueError(f"{aug_name} is not a supported augmentation mode")
    if wrapped:
        return partial(AugmenterWrapper, augmenter=augmenter)
    else:
        return augmenter


class AugmenterWrapper:
    def __init__(self, col, augmenter):
        self.col_reader = transforms.ColReader(col)
        self.augmenter = augmenter

    def __call__(self, *args, **kwargs):
        text = self.col_reader(*args, **kwargs)
        augmented_text = self.augmenter(text)
        return augmented_text


class RuleBasedAugmenter:
    def __init__(self):
        self.augmenter = naw.SynonymAug(aug_src='wordnet')

    def __call__(self, text: str):
        augmented_text = self.augmenter.augment(text)
        return augmented_text


class RandomWordAugmenter:
    """
    https://nlpaug.readthedocs.io/en/latest/augmenter/word/random.html
    """
    def __init__(self, action="swap", *args, **kwargs):
        self.augmenter = naw.RandomWordAug(action=action, *args, **kwargs)

    def __call__(self, text: str):
        augmented_text = self.augmenter.augment(text)
        return augmented_text


class MLMAugmenter(ABC):
    def __init__(self, model_name_or_path=None, tokenizer=None, min_fraction: float = 0.05, max_fraction: float = 0.5,
                 min_mask: int = 1, max_mask: int = 100, topk: int = 10, uniform: bool = False, device=None):
        """
        :param model: huggingface/transformers model for masked language modeling
            e.g model = RobertaForMaskedLM.from_pretrained('roberta-base', return_dict=True)
        :param tokenizer: huggingface/transformers tokenizer
            e.g tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        :param min_fraction: minimum fraction of words to substitute/insert
        :param max_fraction: maximum fraction of words to substitute/insert
        :param min_mask: minimum number of tokens to mask
        :param max_mask: maximum number ot tokens to mask
        :param topk: number of top words to sample from
        :param uniform: whether to sample uniformly from topk words (defaults to False)
        :param device: torch.device
        """
        self.device = device or torch.device('cuda')
        model_name_or_path = model_name_or_path or 'roberta-base'
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, return_dict=True)
        self.model = model.eval().to(self.device)
        # Warning: if a model_name_or_path is a path to a model different than roberta, than default tokenizer will
        # be incompatible
        tokenizer = tokenizer or AutoTokenizer.from_pretrained('roberta-base', use_fast=False)
        self.tokenizer = tokenizer
        self.vocab_words = [self.tokenizer.convert_tokens_to_string(word).strip() for word in tokenizer.get_vocab().keys()]
        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id
        self.topk = topk
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.uniform = uniform
        assert max_fraction >= min_fraction
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

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
        return word

    @abstractmethod
    def __call__(self, text: str):
        pass


class MLMInsertionAugmenter(MLMAugmenter):
    def __call__(self, text: str):
        if self.max_fraction == 0:
            return text
        words = np.array(text.split(), dtype='object')
        max_len = self.tokenizer.model_max_length
        fraction = self.min_fraction + (self.max_fraction - self.min_fraction)*np.random.beta(2, 6.9)
        n_mask = max(self.min_mask, int(min(max_len, len(words)) * fraction))
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
        if self.max_fraction == 0:
            return text
        try:
            words = np.array(text.split(), dtype='object')
            max_len = self.tokenizer.model_max_length
            fraction = self.min_fraction + (self.max_fraction - self.min_fraction) * np.random.beta(2, 6.8)
            n_mask = max(self.min_mask, int(min(max_len, len(words)) * fraction))
            n_mask = min(n_mask, self.max_mask)
            # offset, since lenght might increase after tokenization
            max_masked_idx = min(max_len // 2, len(words) + 1)
            vocab_word_indices = [i for i, word in itertools.islice(enumerate(words), max_masked_idx)
                                  if self.substitute_word(word)]
            if not vocab_word_indices:
                return text
            n_mask = min(n_mask, len(vocab_word_indices))
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
        except Exception as e:
            print(f"Something went wrong during augmentation: {e}")
            print("Text:", text)
            new_text = text
        return new_text

    def substitute_word(self, word):
        # can be later improved to include only some parts of speech etc.
        if word == self.tokenizer.sep_token:
            return False
        return word in self.vocab_words or word[:-1] in self.vocab_words


class BartAugmenter:
    def __init__(self, model=None, tokenizer=None, fraction: float = 0.2, min_mask: int = 1, max_mask: int = 100,
                 lambda_: float = 2.5, num_beams: int = 1, device=None):
        """
        :param model: huggingface/transformers model for masked language modeling
            e.g model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', return_dict=True)
        :param tokenizer: huggingface/transformers tokenizer
            e.g tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        :param fraction: fraction of words to insert
        :param min_mask: minimum number of <mask> tokens to insert
        :param max_mask: maximum number ot tokens to mask
        :param lambda_: mean length of masked subsequence (Poisson distribution)
        :param num_beams: num_beams passed to model.generate()
        :param device: torch.device
        """
        self.device = device or torch.device('cuda')
        model = model or AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large', return_dict=True)
        self.model = model.eval().to(self.device)
        tokenizer = tokenizer or AutoTokenizer.from_pretrained('facebook/bart-large', use_fast=False)
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.mask_token
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.fraction = fraction
        self.lambda_ = lambda_
        self.num_beams = num_beams

    def __call__(self, text: str):
        if self.fraction == 0:
            return text

        words = text.split()
        n_mask = max(self.min_mask, round(len(words) * self.fraction))
        n_mask = min(n_mask, self.max_mask)
        # offset, since lenght might increase after tokenization
        max_masked_idx = min(self.tokenizer.model_max_length - 50, len(words))
        n_places = max(1, round(n_mask / self.lambda_))

        places = np.sort(np.random.choice(max_masked_idx, size=n_places, replace=False))
        lengths = np.random.poisson(self.lambda_, size=n_places)
        ends = {start: start + length for start, length in zip(places, lengths)}
        to_mask = {start + i for start, length in zip(places, lengths) for i in range(length)}

        masked_words = list()
        i = 0
        while i < len(words):
            if i in ends:
                if len(masked_words) == 0 or masked_words[-1] != self.mask_token:
                    masked_words.append(self.mask_token)
                    i = ends[i]
                else:
                    masked_words.append(words[i])
                    i += 1
            elif i in to_mask:
                i += 1
            else:
                masked_words.append(words[i])
                i += 1

        masked_text = " ".join(masked_words)
        inputs = self.tokenizer(masked_text, max_length=1024, truncation=True, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"].to(self.device)

        # Generate seq2seq output
        with torch.no_grad():
            summary_ids = \
                self.model.generate(input_ids, num_beams=self.num_beams, max_length=512, early_stopping=True)[0]
            # 2 in indexing is a magic number
            generated_text = self.tokenizer.decode(summary_ids[2:],
                                                   skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return generated_text


class NoAugmenter:
    def __call__(self, text: str):
        return text
