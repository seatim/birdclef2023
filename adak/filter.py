
"""Functions for post-processing inferences.
"""

import math

import numpy as np


def top_k_filter(preds, k, assert_input_is_normalized=False):
    """Return modified copy of ``preds``.  For each prediction in ``preds``,
    keep the K highest scores and set the rest to zero.  Then scale the
    remaining scores so they sum to one.

    Args:
        preds (array): 2D NumPY array
        k (int): the K in "K highest"

    Returns:
        2D NumPY array

    """

    def top_k_filter(pred):
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

    return np.stack([top_k_filter(pred) for pred in np.array(preds)])


def fine_filter(preds, threshold, assert_input_is_normalized=False):
    """Return modified copy of ``preds``.  For each prediction in ``preds``,
    set scores that are lower than ``threshold`` to zero.  Then scale the
    remaining scores so they sum to one.

    Args:
        preds (array): 2D NumPY array
        threshold (float)

    Returns:
        2D NumPY array

    """
    if not (0 < threshold < 1):
        raise ValueError('threshold must be between zero and one')

    def fine_filter(pred):
        if assert_input_is_normalized:
            assert math.isclose(sum(pred), 1, rel_tol=1e-5), sum(pred)

        new_pred = np.where(pred < threshold, 0, pred)
        if sum(new_pred):
            new_pred /= sum(new_pred)
            assert math.isclose(sum(new_pred), 1, rel_tol=1e-6), sum(new_pred)
        return new_pred

    return np.stack([fine_filter(pred) for pred in np.array(preds)])


def max_filter(preds, threshold):
    """Return modified copy of ``preds``.  For each prediction in ``preds``,
    if the highest score in the prediction vector is less than ``threshold``
    set the vector to zero.

    Args:
        preds (array): 2D NumPY array
        threshold (float)

    Returns:
        2D NumPY array

    """
    return np.stack([(np.zeros(len(pred)) if max(pred) < threshold else pred)
                     for pred in np.array(preds)])
