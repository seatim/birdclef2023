
import unittest

import numpy as np

from adak.filter import top_k_filter, fine_filter


class Test_top_k_filter(unittest.TestCase):
    def test1(self):
        preds = [[0, 0.1, 0.9], [0.1, 0.7, 0.2]]
        self.assertEqual(top_k_filter(preds, 3).tolist(), preds)

        expected_2 = [[0, 0.1, 0.9], [0, 7/9, 2/9]]
        self.assertTrue(np.isclose(top_k_filter(preds, 2), expected_2).all())

        expected_1 = [[0, 0, 1], [0, 1, 0]]
        self.assertEqual(top_k_filter(preds, 1).tolist(), expected_1)

    def test_assert_input_is_normalized(self):
        preds = [[0, 0.1, 0.9], [0.1, 0.7, 0.3]]
        with self.assertRaises(AssertionError):
            top_k_filter(preds, 3, True)


class Test_fine_filter(unittest.TestCase):
    def test1(self):
        preds = [[0, 0.1, 0.9], [0.1, 0.7, 0.2]]
        self.assertEqual(fine_filter(preds, 0.09).tolist(), preds)

        expected_2 = [[0, 0, 1], [0, 7/9, 2/9]]
        self.assertTrue(np.isclose(
            fine_filter(preds, 0.19), expected_2).all())

        expected_1 = [[0, 0, 1], [0, 1, 0]]
        self.assertEqual(fine_filter(preds, 0.5).tolist(), expected_1)

        expected_0 = [[0, 0, 1], [0, 0, 0]]
        self.assertEqual(fine_filter(preds, 0.9).tolist(), expected_0)

    def test_assert_input_is_normalized(self):
        preds = [[0, 0.1, 0.9], [0.1, 0.7, 0.3]]
        with self.assertRaises(AssertionError):
            fine_filter(preds, 0.9, True)
