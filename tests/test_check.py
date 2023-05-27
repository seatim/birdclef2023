
import unittest

from os.path import dirname, join
from unittest.mock import patch

from adak.check import check_image, check_images
from adak.config import TrainConfig

IMAGES_DIR = join(dirname(__file__), 'data', 'train_images')
GOOD_IMAGE = join(IMAGES_DIR, 'helgui', 'XC503001.ogg-0-3.png')


class Test_check_image(unittest.TestCase):
    def test1(self):
        self.assertIsNone(check_image(TrainConfig, GOOD_IMAGE))

    def test_truncated_image_file(self):
        img_path = join(IMAGES_DIR, '..', 'truncated.png')
        self.assertIn('Truncated', check_image(TrainConfig, img_path))

    def test_not_an_image_file(self):
        img_path = join(IMAGES_DIR, '..', 'train_metadata.csv')
        self.assertIn('cannot identify', check_image(TrainConfig, img_path))

    def test_bad_shape(self):
        config = TrainConfig.from_dict(n_mels=100)
        self.assertIn('image height !=', check_image(config, GOOD_IMAGE))


class Test_check_images(unittest.TestCase):
    @patch('builtins.print')
    def test1(self, print):
        config = TrainConfig.from_dict(combined_images_dir=IMAGES_DIR)
        classes = check_images(config, True)
        self.assertEqual(classes, {'helgui', 'subbus1'})
        self.assertEqual(print.call_count, 5)
