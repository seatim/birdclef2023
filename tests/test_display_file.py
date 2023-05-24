
import contextlib
import io
import os
import unittest
import tempfile

from os.path import dirname, join
from unittest.mock import patch

from adak.config import BaseConfig
from display_file import main

TEST_AUDIO_PATH = join(dirname(__file__), 'data', 'XC503001.ogg')


class Test_display_file(unittest.TestCase):
    def setUp(self):
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        for name in os.listdir(self.out_dir):
            os.remove(join(self.out_dir, name))

        os.rmdir(self.out_dir)
        self.out_dir = None

    def test1(self):
        f = io.StringIO()

        with contextlib.redirect_stdout(f):
            try:
                main([TEST_AUDIO_PATH])
            except SystemExit:
                pass

        self.assertIn('Sample rate: 32000', f.getvalue())
        self.assertIn('Num. samples: 2586331', f.getvalue())

    def test2(self):
        f = io.StringIO()

        with contextlib.redirect_stdout(f):
            try:
                main([TEST_AUDIO_PATH, '-XYZH', '-o', self.out_dir])
            except SystemExit:
                pass

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
        self.assertEqual(os.listdir(self.out_dir), created_files)

    @patch('matplotlib.pyplot.show')
    def test3(self, plt_show):
        f = io.StringIO()

        with contextlib.redirect_stdout(f):
            try:
                main([TEST_AUDIO_PATH, '-xyzw', '-l', '12.3'])
            except SystemExit:
                pass

        self.assertEqual(plt_show.call_count, 4)
