
import math

import numpy as np

from torch import tensor

from .glue import avg_precision


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


def do_filter_top_k(preds, k, assert_input_is_normalized=False):
    def filter_top_k(pred):
        if assert_input_is_normalized:
            assert math.isclose(sum(pred), 1, rel_tol=1e-5), sum(pred)

        pred_argsort = pred.argsort()
        top_k_indices = pred_argsort[-k:]
        bottom_n_k_indices = pred_argsort[:-k]
        top_k_values = pred[top_k_indices]

        new_pred = pred.copy()
        new_pred[top_k_indices] = top_k_values / sum(top_k_values)
        new_pred[bottom_n_k_indices] = 0
        assert math.isclose(sum(new_pred), 1, rel_tol=1e-6), sum(new_pred)
        return new_pred

    return np.stack([filter_top_k(pred) for pred in np.array(preds)])


def fine_threshold(preds, threshold, assert_input_is_normalized=False):
    if not (0 < threshold < 1):
        raise ValueError('threshold must be between zero and one')

    def fine_threshold(pred):
        if assert_input_is_normalized:
            assert math.isclose(sum(pred), 1, rel_tol=1e-5), sum(pred)

        new_pred = np.where(pred < threshold, 0, pred)
        if sum(new_pred):
            new_pred /= sum(new_pred)
            assert math.isclose(sum(new_pred), 1, rel_tol=1e-6), sum(new_pred)
        return new_pred

    return np.stack([fine_threshold(pred) for pred in np.array(preds)])


def calculate_n_top_n(y_pred, y_true, classes, n):
    n_samples = _validate_y_args(y_pred, y_true, classes)

    top_n = [pred.argsort()[-n:] for pred in y_pred]
    assert len(top_n) == n_samples, (len(top_n), n_samples)

    return sum(true_k in top_n_k for true_k, top_n_k in zip(y_true, top_n))
