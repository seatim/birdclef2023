
import numpy as np

from fastai.vision.all import L, parent_label
from sklearn.model_selection import train_test_split

from .evaluate import avg_precision  # part of this module's API now


class StratifiedSplitter:
    def __init__(self, test_size, random_state, print_stats=True):
        self.test_size = test_size
        self.random_state = random_state
        self.print_stats = print_stats

    def __call__(self, items):
        labels = np.array([parent_label(item) for item in items])
        train, valid = train_test_split(
            list(range(len(items))), test_size=self.test_size,
            random_state=self.random_state, stratify=labels, train_size=None,
            shuffle=True)

        assert set(train) & set(valid) == set()
        assert len(set(train) | set(valid)) == len(items)

        if self.print_stats:
            print('I: len(train) =', len(train))
            print('I: len(valid) =', len(valid))
            print('I: n. classes in train =', len(set(labels[train])))
            print('I: n. classes in valid =', len(set(labels[valid])))

        return L(train), L(valid)
