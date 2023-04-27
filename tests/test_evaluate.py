
import unittest

from adak.evaluate import avg_precision_over_subset as apos


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
