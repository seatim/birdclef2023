
import os
import shutil
import tempfile
import unittest

from os.path import join

import pandas as pd

from evaluate import main as evaluate

from . import run_main
from .test_train_classifier import IMAGES_DIR, PRETRAINED_MODEL_PATH


class Test_evaluate(unittest.TestCase):
    def setUp(self):
        self.preds_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.preds_dir)
        self.preds_dir = None

    def test1(self):
        args = ['-S', IMAGES_DIR, PRETRAINED_MODEL_PATH]
        output = run_main(evaluate, args)
        self.assertIn('20 inferences', output)

    def test_ensemble(self):
        args = ['-S', IMAGES_DIR, PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_PATH]
        output = run_main(evaluate, args)
        self.assertIn('20 inferences', output)

    def test_ensemble_efficient(self):
        args = ['-S', IMAGES_DIR, PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_PATH,
                '-t', '0.5']
        output = run_main(evaluate, args)
        self.assertIn('20 inferences', output)

    def test_save_preds(self):
        preds_path = join(self.preds_dir, 'preds.csv')
        args = ['-S', IMAGES_DIR, PRETRAINED_MODEL_PATH, '-s', preds_path]
        output = run_main(evaluate, args)
        self.assertIn('20 inferences', output)

        self.assertEqual(os.listdir(self.preds_dir), ['preds.csv'])
        df = pd.read_csv(preds_path, index_col=0)
        self.assertEqual(list(df.columns), ['path', 'helgui', 'subbus1'])
        self.assertEqual(len(df.index), 20)
