
import os
import tempfile
import unittest

from os.path import dirname, join
from unittest.mock import patch

import pandas as pd

from adak.config import BaseConfig
from analyze_preds import main as analyze_preds

from . import run_main
from .test_transform import MaskWarnings

TEST_DATA_DIR = join(dirname(__file__), 'data')
PREDS_PATH = join(TEST_DATA_DIR, 'preds', 'model-1.1-preds.csv')
BaseConfig.audio_dir = join(TEST_DATA_DIR, 'train_audio')


class Test_analyze_preds(MaskWarnings):
    def setUp(self):
        super().setUp()
        _, self.temp_path = tempfile.mkstemp()

    def tearDown(self):
        super().tearDown()
        os.remove(self.temp_path)

    def test1(self):
        output = run_main(analyze_preds, [PREDS_PATH])
        self.assertIn('20 inferences', output)

    def test_show_stats(self):
        output = run_main(analyze_preds, [PREDS_PATH, '-S'])
        self.assertIn('20 inferences', output)
        self.assertIn('Statistics of sum', output)
        self.assertIn('Statistics of max', output)

    @patch('matplotlib.pyplot.show')
    def test_show_stats_and_hist(self, plt_show):
        output = run_main(analyze_preds, [PREDS_PATH, '-Ss'])
        self.assertIn('20 inferences', output)
        self.assertIn('Statistics of sum', output)
        self.assertIn('Statistics of max', output)
        self.assertEqual(plt_show.call_count, 2)

    def test_report_sweeps(self):
        output = run_main(analyze_preds, [PREDS_PATH, '-r'])
        self.assertIn('20 inferences', output)
        self.assertIn('AP scores', output)

    def test_report_class_stats(self):
        output = run_main(analyze_preds, [PREDS_PATH, '-R'])
        self.assertIn('20 inferences', output)
        self.assertIn('Highest confidence classes', output)
        self.assertIn('Lowest confidence classes', output)
        self.assertIn('Statistics of max', output)

    def test_threshold(self):
        output = run_main(analyze_preds, [PREDS_PATH, '-p', '0.8'])
        self.assertIn('Examples with max', output)
        self.assertIn('Top five predictions', output)

    def test_list_nse_candidates(self):
        output = run_main(analyze_preds, [PREDS_PATH, '-e'])
        self.assertIn('Examples with lowest sum', output)
        self.assertIn('Examples with lowest max', output)

    def test_make_nse_file(self):
        temp_path = self.temp_path
        output = run_main(analyze_preds, [PREDS_PATH, '-m', '-n', temp_path])
        df = pd.read_csv(temp_path)
        self.assertEqual(list(df.columns), ['max_bc23', 'path', 'sha1'])
        self.assertEqual(len(df.index), 20)
