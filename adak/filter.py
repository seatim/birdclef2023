
import math

import numpy as np


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


def sum_filter(preds, threshold):
    return np.stack([(np.zeros(len(pred)) if sum(pred) < threshold else pred)
                     for pred in np.array(preds)])


def max_filter(preds, threshold):
    return np.stack([(np.zeros(len(pred)) if max(pred) < threshold else pred)
                     for pred in np.array(preds)])
