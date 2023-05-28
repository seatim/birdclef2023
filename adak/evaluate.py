
"""Functions for measuring inference performance.
"""

import numpy as np

from sklearn.metrics import average_precision_score
from torch import tensor
from torch.nn.functional import one_hot


def avg_precision(y_pred, y_true, n_classes):
    """Adapter for `sklearn.metrics.average_precision_score` that works for
    multi-class classification.

    Args:
        y_pred (tensor): a tensor of shape (N, M) where N is number of examples
            and M is number of classes, and values are floats in the range
            [0..1]

        y_true (tensor): a tensor of shape (N,) with int values

        n_classes (int): number of classes

    Returns:
        average precision score (float)

    """
    if len(y_pred.shape) != 2:
        raise ValueError('y_pred must be a rank-2 tensor')
    if len(y_true.shape) != 1:
        raise ValueError('y_true must be a rank-1 tensor')
    if not (0 <= y_pred.min() <= y_pred.max() <= 1):
        raise ValueError('y_pred values must be between 0 and 1')
    if not (0 <= min(y_true) <= max(y_true) < n_classes):
        raise ValueError('y_true values must be between 0 and n_classes')

    assert y_pred.shape[1] == n_classes, y_pred.shape
    return average_precision_score(one_hot(y_true, n_classes), y_pred)


def _validate_y_args(y_pred, y_true, classes, n_samples=None):
    """Check the validity of ``y_pred`` and ``y_true``.

    Args:
        y_pred (tensor/array): a tensor or NumPY array of shape (N, M) where N
            is number of examples and M is number of classes, and values are
            floats in the range [0..1]

        y_true (tensor/array): a tensor or NumPY array of shape (N,) with int
            values

        classes (sequence)

        n_samples (int/None): number of samples

    Returns:
        number of samples

    """
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
    """Return matched slices of ``y_pred`` and ``y_true``.

    Args:
        y_pred (tensor/array): a tensor or NumPY array of shape (N, M) where N
            is number of examples and M is number of classes, and values are
            floats in the range [0..1]

        y_true (tensor/array): a tensor or NumPY array of shape (N,) with int
            values

        classes (sequence)

        subset (iterable): iterable whose elements are in ``classes``

    Returns:
        A 2-tuple comprising:

            - a NumPY array of shape (N, S) whose columns are columns of
              ``y_pred``, where S is the length of ``subset``

            - a NumPY array of shape (N,) whose values are mapped from
              ``y_true`` to correspond to ``subset``

    """
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
    """Calculate average precision score of slice of ``y_pred``.

    Args:
        y_pred (tensor/array): a tensor or NumPY array of shape (N, M) where N
            is number of examples and M is number of classes, and values are
            floats in the range [0..1]

        y_true (tensor/array): a tensor or NumPY array of shape (N,) with int
            values

        classes (sequence)

        subset (iterable): iterable whose elements are in ``classes``

    Returns:
        average precision score (float)

    """
    y_pred, y_true = slice_by_class_subset(y_pred, y_true, classes, subset)
    return avg_precision(y_pred, tensor(y_true), len(subset))


def count_top_n(y_pred, y_true, classes, n):
    """Count the number of predictions in ``y_pred`` for which the true class
    received one of the N highest scores.

    Args:
        y_pred (tensor/array): a tensor or NumPY array of shape (L, M) where L
            is number of examples and M is number of classes, and values are
            floats in the range [0..1]

        y_true (tensor/array): a tensor or NumPY array of shape (L,) with int
            values

        classes (sequence)

        n (int): the N in "N highest"

    Returns:
        number of top-N predictions (int)

    """
    n_samples = _validate_y_args(y_pred, y_true, classes)

    top_n = [pred.argsort()[-n:] for pred in y_pred]
    assert len(top_n) == n_samples, (len(top_n), n_samples)

    return sum(true_k in top_n_k for true_k, top_n_k in zip(y_true, top_n))
