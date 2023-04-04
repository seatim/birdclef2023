
import unittest
import warnings

from os.path import dirname, join

from adak.transform import image_from_audio, image_width, DEFAULT_N_MELS

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
