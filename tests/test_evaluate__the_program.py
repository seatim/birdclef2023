
import unittest

from evaluate import main as evaluate

from . import run_main
from .test_train_classifier import IMAGES_DIR, PRETRAINED_MODEL_PATH


class Test_evaluate(unittest.TestCase):
    def test1(self):
        args = ['-S', IMAGES_DIR, PRETRAINED_MODEL_PATH]
        output = run_main(evaluate, args)

    def test_ensemble(self):
        args = ['-S', IMAGES_DIR, PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_PATH]
        output = run_main(evaluate, args)

    def test_ensemble_efficient(self):
        args = ['-S', IMAGES_DIR, PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_PATH,
                '-t', '0.5']
        output = run_main(evaluate, args)
