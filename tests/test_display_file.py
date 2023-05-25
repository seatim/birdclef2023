
import os
import unittest
import shutil
import tempfile

from unittest.mock import patch

from adak.config import BaseConfig
from display_file import main as display_file

from . import run_main
from .test_transform import TEST_AUDIO_PATH


class Test_display_file(unittest.TestCase):
    def setUp(self):
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.out_dir)
        self.out_dir = None

    def test1(self):
        args = [TEST_AUDIO_PATH]
        output = run_main(display_file, args)

        self.assertIn('Sample rate: 32000', output)
        self.assertIn('Num. samples: 2586331', output)

    def test2(self):
        args = [TEST_AUDIO_PATH, '-XYZH', '-o', self.out_dir]
        output = run_main(display_file, args)

        n_mels = BaseConfig.n_mels
        n_fft = BaseConfig.n_fft
        created_files = [f'XC503001.ogg.mel{n_fft}_{n_mels}.cliptails.png',
                         f'XC503001.ogg.mel{n_fft}_{n_mels}.rescale.png',
                         f'XC503001.ogg.mel{n_fft}_{n_mels}.histeq.png',
                         f'XC503001.ogg.mel{n_fft}_{n_mels}.adapteq.png',
                         f'XC503001.ogg.mel{n_fft}_{n_mels}.centmed.png',
                         f'XC503001.ogg.phase{n_fft}.png',
                         f'XC503001.ogg.mag{n_fft}.png',
                         f'XC503001.ogg.mel{n_fft}_{n_mels}.png']
        self.assertEqual(set(os.listdir(self.out_dir)), set(created_files))

    @patch('matplotlib.pyplot.show')
    def test3(self, plt_show):
        args = [TEST_AUDIO_PATH, '-xyzw', '-l', '12.3']
        output = run_main(display_file, args)

        self.assertEqual(plt_show.call_count, 4)
