
import unittest

import soundfile

from adak.config import BaseConfig
from adak.sed import SoundEventDetectionFilter, sound_event_proba

from . import capture_stdout
from .test_transform import TEST_AUDIO_PATH
from .test_train_classifier import NSE_FILE_PATH


class TestSoundEventDetectionFilter(unittest.TestCase):
    def test1(self):
        sed = SoundEventDetectionFilter(NSE_FILE_PATH, NSE_FILE_PATH)
        self.assertIsNone(sed.threshold)

        def set_threshold(thr):
            sed.threshold = thr

        output = capture_stdout(set_threshold, 0.199)
        self.assertEqual(sed.threshold, 0.199)

        output = capture_stdout(sed.examine)
        self.assertIn('20 unique hashes', output)


class Test_sound_event_proba(unittest.TestCase):
    def test1(self):
        audio, sr = soundfile.read(TEST_AUDIO_PATH)
        probs = sound_event_proba(audio, BaseConfig)
        self.assertEqual(len(probs), 33)
        self.assertTrue(all(0 <= p <= 1 for p in probs))
