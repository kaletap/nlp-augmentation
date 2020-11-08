class NoAug:
    # just column reader
    def __init__(self, col_name):
        self.col_name = col_name

    def __call__(self, x, *args, **kwargs):
        return x[self.col_name]


class RuleBasedAugmenter:
    pass
