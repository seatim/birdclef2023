
import unittest

import numpy as np

from adak.evaluate import (avg_precision_over_subset as apos,
                           do_filter_top_k as dftk, fine_threshold)


class Test_apos(unittest.TestCase):
    def test1(self):
        y_pred = [[0.1, 0, 0.9, 0], [0.2, 0, 0.8, 0]]
        y_true = [2, 0]
        actual = float(apos(y_pred, y_true, list('abcd'), list('ac')))
        expected = 1.0
        self.assertEqual(actual, expected)

    def test2(self):
        y_pred = [[0, 0, 0.9, 0], [0.2, 0, 0.8, 0]]
        y_true = [0, 2]
        actual = float(apos(y_pred, y_true, list('abcd'), list('ac')))
        expected = 0.5
        self.assertEqual(actual, expected)


class Test_dftk(unittest.TestCase):
    def test1(self):
        preds = [[0, 0.1, 0.9], [0.1, 0.7, 0.2]]
        self.assertEqual(dftk(preds, 3).tolist(), preds)

        expected_2 = [[0, 0.1, 0.9], [0, 7/9, 2/9]]
        self.assertTrue(np.isclose(dftk(preds, 2), expected_2).all())

        expected_1 = [[0, 0, 1], [0, 1, 0]]
        self.assertEqual(dftk(preds, 1).tolist(), expected_1)

    def test_assert_input_is_normalized(self):
        preds = [[0, 0.1, 0.9], [0.1, 0.7, 0.3]]
        with self.assertRaises(AssertionError):
            dftk(preds, 3, True)


class Test_fine_threshold(unittest.TestCase):
    def test1(self):
        preds = [[0, 0.1, 0.9], [0.1, 0.7, 0.2]]
        self.assertEqual(fine_threshold(preds, 0.09).tolist(), preds)

        expected_2 = [[0, 0, 1], [0, 7/9, 2/9]]
        self.assertTrue(np.isclose(
            fine_threshold(preds, 0.19), expected_2).all())

        expected_1 = [[0, 0, 1], [0, 1, 0]]
        self.assertEqual(fine_threshold(preds, 0.5).tolist(), expected_1)

        expected_0 = [[0, 0, 1], [0, 0, 0]]
        self.assertEqual(fine_threshold(preds, 0.9).tolist(), expected_0)

    def test_assert_input_is_normalized(self):
        preds = [[0, 0.1, 0.9], [0.1, 0.7, 0.3]]
        with self.assertRaises(AssertionError):
            fine_threshold(preds, 0.9, True)
