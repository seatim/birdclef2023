
import numpy as np

from fastai.vision.all import L, parent_label
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot


def avg_precision(y_pred, y_true, n_classes):
    assert y_pred.shape[1] == n_classes, y_pred.shape
    return average_precision_score(one_hot(y_true, n_classes), y_pred)


class StratifiedSplitter:
    def __init__(self, test_size, random_state):
        self.test_size = test_size
        self.random_state = random_state

    def __call__(self, items, **kwargs):
        if kwargs:
            print('W: StratifiedSplitter: unknown kwargs:', kwargs)

        labels = np.array([parent_label(item) for item in items])
        train, valid = train_test_split(
            list(range(len(items))), test_size=self.test_size,
            random_state=self.random_state, stratify=labels, train_size=None,
            shuffle=True)

        assert set(train) & set(valid) == set()
        assert len(set(train) | set(valid)) == len(items)
        print('I: len(train) =', len(train))
        print('I: len(valid) =', len(valid))
        print('I: n. classes in train =', len(set(labels[train])))
        print('I: n. classes in valid =', len(set(labels[valid])))

        return L(train), L(valid)
