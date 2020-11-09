from fastai.data import transforms


class NoAug:
    def __init__(self, col_name):
        self.reader = transforms.ColReader(col_name)

    def __call__(self, *args, **kwargs):
        return self.reader(*args, **kwargs)


class RuleBasedAugmenter:
    pass
