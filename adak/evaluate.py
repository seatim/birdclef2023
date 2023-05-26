
"""Functions for measuring inference performance.
"""

import numpy as np

from sklearn.metrics import average_precision_score
from torch import tensor
from torch.nn.functional import one_hot


def avg_precision(y_pred, y_true, n_classes):
    assert y_pred.shape[1] == n_classes, y_pred.shape
    return average_precision_score(one_hot(y_true, n_classes), y_pred)


def _validate_y_args(y_pred, y_true, classes, n_samples=None):
    assert len(y_pred.shape) == 2, y_pred.shape
    assert len(y_true.shape) == 1, y_true.shape

    if n_samples is None:
        n_samples = y_pred.shape[0]
    else:
        assert n_samples == y_pred.shape[0], (n_samples, y_pred.shape)

    assert n_samples == len(y_true), (n_samples, len(y_true))
    assert y_pred.shape[1] == len(classes), (y_pred.shape, len(classes))
    assert all(idx in range(len(classes)) for idx in y_true)

    return n_samples


def slice_by_class_subset(y_pred, y_true, classes, subset):
    unknown_classes = set(subset) - set(classes)
    assert unknown_classes == set(), unknown_classes
    assert hasattr(classes, '__getitem__'), 'classes must be ordered'

    classes = list(classes)
    subset = list(subset)
    subset_indices = [classes.index(label) for label in subset]

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    n_samples = _validate_y_args(y_pred, y_true, classes)

    y_pred = np.array([pred[subset_indices] for pred in y_pred])
    y_true = np.array([subset_indices.index(idx) for idx in y_true])
    _validate_y_args(y_pred, y_true, subset, n_samples)

    return y_pred, y_true


def avg_precision_over_subset(y_pred, y_true, classes, subset):
    y_pred, y_true = slice_by_class_subset(y_pred, y_true, classes, subset)
    return avg_precision(y_pred, tensor(y_true), len(subset))


def calculate_n_top_n(y_pred, y_true, classes, n):
    n_samples = _validate_y_args(y_pred, y_true, classes)

    top_n = [pred.argsort()[-n:] for pred in y_pred]
    assert len(top_n) == n_samples, (len(top_n), n_samples)

    return sum(true_k in top_n_k for true_k, top_n_k in zip(y_true, top_n))
