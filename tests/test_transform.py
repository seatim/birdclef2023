
import math
import unittest
import warnings

from os.path import dirname, join

import numpy as np

from parameterized import parameterized

from adak.transform import (image_from_audio, image_width, images_from_audio,
                            DEFAULT_N_MELS)

TEST_AUDIO_PATH = join(dirname(__file__), 'data', 'XC503001.ogg')
TEST_AUDIO_PLAY_TIME = 80.822


class MaskWarnings(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings(
            action='ignore', category=DeprecationWarning,
            message='pkg_resources is deprecated as an API')
        warnings.filterwarnings(
            action='ignore', category=DeprecationWarning,
            message='Deprecated call to `pkg_resources.declare_namespace')


class Test_image_from_audio(MaskWarnings):
    def test1(self):
        img = image_from_audio(TEST_AUDIO_PATH)
        expected_img_width = image_width(TEST_AUDIO_PLAY_TIME)
        self.assertEqual(img.shape, (DEFAULT_N_MELS, expected_img_width))


class Test_images_from_audio(MaskWarnings):
    @parameterized.expand(((3,), (5,), (11.1,)))
    def test1(self, max_play_time):
        imgs = np.array(images_from_audio(TEST_AUDIO_PATH, max_play_time))

        orig_img_width = image_width(TEST_AUDIO_PLAY_TIME)
        sub_img_width = image_width(max_play_time)
        expected_n_imgs = math.ceil(orig_img_width / sub_img_width)

        self.assertEqual(imgs.shape,
            (expected_n_imgs, DEFAULT_N_MELS, sub_img_width))
