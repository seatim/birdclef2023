
import math

import numpy as np

from torch import tensor

from .glue import avg_precision


def avg_precision_over_subset(y_pred, y_true, classes, subset):
    unknown_classes = set(subset) - set(classes)
    assert unknown_classes == set(), unknown_classes

    classes = list(classes)
    subset = list(subset)
    subset_indices = [classes.index(label) for label in subset]

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    assert len(y_pred.shape) == 2, y_pred.shape
    assert len(y_true.shape) == 1, y_true.shape
    n_samples = y_pred.shape[0]
    assert n_samples == len(y_true), (n_samples, len(y_true))
    assert y_pred.shape[1] == len(classes), (y_pred.shape, len(classes))
    assert all(idx in range(len(classes)) for idx in y_true)

    y_pred = np.array([pred[subset_indices] for pred in y_pred])
    y_true = np.array([subset_indices.index(idx) for idx in y_true])

    assert len(y_pred.shape) == 2, y_pred.shape
    assert len(y_true.shape) == 1, y_true.shape
    assert n_samples == len(y_true), (n_samples, len(y_true))
    assert n_samples == y_pred.shape[0], (n_samples, y_pred.shape)
    assert y_pred.shape[1] == len(subset), (y_pred.shape, len(subset))
    assert all(idx in range(len(subset)) for idx in y_true)

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


def apply_threshold(preds, threshold, assert_input_is_normalized=False):
    if not (0 < threshold < 1):
        raise ValueError('threshold must be between zero and one')

    def apply_threshold(pred):
        if assert_input_is_normalized:
            assert math.isclose(sum(pred), 1, rel_tol=1e-5), sum(pred)

        new_pred = np.where(pred < threshold, 0, pred)
        if sum(new_pred):
            new_pred /= sum(new_pred)
            assert math.isclose(sum(new_pred), 1, rel_tol=1e-6), sum(new_pred)
        return new_pred

    return np.stack([apply_threshold(pred) for pred in np.array(preds)])
