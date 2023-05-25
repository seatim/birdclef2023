
import math
import unittest
import warnings

from itertools import product
from os.path import dirname, join

import numpy as np

from parameterized import parameterized

from adak.config import BaseConfig
from adak.transform import image_from_audio, images_from_audio

AUDIO_DIR = join(dirname(__file__), 'data', 'train_audio')
TEST_AUDIO_PATH = join(AUDIO_DIR, 'helgui', 'XC503001.ogg')
TEST_AUDIO_PLAY_TIME = 80.822


class MaskWarnings(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings(
            action='ignore', category=DeprecationWarning,
            message='pkg_resources is deprecated as an API')
        warnings.filterwarnings(
            action='ignore', category=DeprecationWarning,
            message='Deprecated call to `pkg_resources.declare_namespace')

        self.config = BaseConfig()


class RealFrameHopFactor(BaseConfig):
    def __init__(self, frame_duration, factor):
        self.frame_duration = frame_duration
        self.factor = factor

    @property
    def frame_hop_length(self):
        return int(self.image_width(self.frame_duration) / self.factor)


class BadFrameHopLength(BaseConfig):
    @property
    def frame_hop_length(self):
        return self.image_width(self.frame_duration) + 1


class Test_image_from_audio(MaskWarnings):
    def test1(self):
        img = image_from_audio(TEST_AUDIO_PATH, self.config)
        expected_img_width = self.config.image_width(TEST_AUDIO_PLAY_TIME)
        self.assertEqual(img.shape, (self.config.n_mels, expected_img_width))


class Test_images_from_audio(MaskWarnings):
    @parameterized.expand(product((3, 5., 11.1, 81.1), (1, 2, 2.5)))
    def test1(self, frame_duration, oversample_factor):
        self.config = RealFrameHopFactor(frame_duration, oversample_factor)

        imgs = np.array(images_from_audio(TEST_AUDIO_PATH, self.config))
        img_width = self.config.image_width(TEST_AUDIO_PLAY_TIME)
        expected_n_imgs = math.ceil(img_width / self.config.frame_hop_length)

        self.assertEqual(imgs.shape,
            (expected_n_imgs, self.config.n_mels, self.config.frame_width))

    def test_bad_frame_hop_length(self):
        with self.assertRaises(ValueError):
            np.array(images_from_audio(TEST_AUDIO_PATH, BadFrameHopLength()))

    @parameterized.expand((1, 2, 3))
    def test_max_frames(self, max_frames):
        imgs = np.array(images_from_audio(
            TEST_AUDIO_PATH, self.config, max_frames))
        self.assertEqual(len(imgs), max_frames)
