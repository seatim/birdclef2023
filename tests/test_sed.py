
import unittest

from os.path import dirname, join

import soundfile

from adak.config import BaseConfig
from adak.sed import sound_event_proba

TEST_AUDIO_PATH = join(dirname(__file__), 'data', 'XC503001.ogg')


class Test_sound_event_proba(unittest.TestCase):
    def test1(self):
        audio, sr = soundfile.read(TEST_AUDIO_PATH)
        probs = sound_event_proba(audio, BaseConfig)
        self.assertEqual(len(probs), 33)
        self.assertTrue(all(0 <= p <= 1 for p in probs))
