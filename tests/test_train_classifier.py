
import shutil
import tempfile
import unittest
import warnings

from os.path import dirname, join

from train_classifier import main as train_classifier

from . import run_main

TEST_DATA_DIR = join(dirname(__file__), 'data')
IMAGES_DIR = join(TEST_DATA_DIR, 'train_images')
PRETRAINED_MODEL_PATH = join(TEST_DATA_DIR, 'models', 'birdclef-model-1.1.pkl')
NSE_FILE_PATH = join(TEST_DATA_DIR, 'preds', 'model-1.1-preds.csv.nsedata.csv')


class Test_train_classifier(unittest.TestCase):
    def setUp(self):
        self.combined_dir = tempfile.mkdtemp('train')
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

    def test_pretrained_model(self):
        args = ['-i', IMAGES_DIR, '-B', '', '-D', '', '-e', '1', '-C',
                '-I', self.combined_dir, '-P', PRETRAINED_MODEL_PATH]
        output = run_main(train_classifier, args)

    def test_nse_file(self):
        args = ['-i', IMAGES_DIR, '-B', '', '-D', '', '-e', '1', '-C',
                '-I', self.combined_dir, '-N', NSE_FILE_PATH, '-t', '0.3']
        output = run_main(train_classifier, args)
