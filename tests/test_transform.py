
import math
import unittest
import warnings

from os.path import dirname, join

import numpy as np

from parameterized import parameterized

from adak.config import TrainConfig
from adak.transform import image_from_audio, image_width, images_from_audio

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

        self.config = TrainConfig()

    def image_width(self, audio_play_time):
        return image_width(audio_play_time, self.config.sample_rate,
                           self.config.hop_length)


class Test_image_from_audio(MaskWarnings):
    def test1(self):
        img = image_from_audio(TEST_AUDIO_PATH, self.config)
        expected_img_width = self.image_width(TEST_AUDIO_PLAY_TIME)
        self.assertEqual(img.shape, (self.config.n_mels, expected_img_width))


class Test_images_from_audio(MaskWarnings):
    @parameterized.expand(((3,), (5,), (11.1,)))
    def test1(self, frame_duration):
        self.config.frame_duration = frame_duration

        imgs = np.array(images_from_audio(TEST_AUDIO_PATH, self.config))
        orig_img_width = self.image_width(TEST_AUDIO_PLAY_TIME)
        frame_img_width = self.image_width(frame_duration)
        expected_n_imgs = math.ceil(orig_img_width / frame_img_width)

        self.assertEqual(imgs.shape,
            (expected_n_imgs, self.config.n_mels, frame_img_width))
