
import shutil
import tempfile
import unittest
import warnings

from os.path import dirname, join

from train_classifier import main as train_classifier

from . import run_main

IMAGES_DIR = join(dirname(__file__), 'data', 'train_images')


class Test_train_classifier(unittest.TestCase):
    def setUp(self):
        self.combined_dir = tempfile.mkdtemp()
        warnings.filterwarnings(
            action='ignore', category=UserWarning,
            message='Your generator is empty.')

    def tearDown(self):
        shutil.rmtree(self.combined_dir)
        self.combined_dir = None

    def test1(self):
        args = ['-i', IMAGES_DIR, '-B', '', '-D', '', '-e', '1', '-C',
                '-I', self.combined_dir]
        output = run_main(train_classifier, args)
