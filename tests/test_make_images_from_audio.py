
import contextlib
import io
import os
import re
import shutil
import tempfile
import unittest

from os.path import join

from adak.config import MakeImagesConfig
from make_images_from_audio import main

from .test_transform import MaskWarnings, AUDIO_DIR


class Test_make_images_from_audio(MaskWarnings):
    def setUp(self):
        super().setUp()

        self.train_dir = tempfile.mkdtemp('train')
        self.val_dir = re.sub('train', 'val', self.train_dir)
        os.mkdir(self.val_dir)

    def tearDown(self):
        super().tearDown()

        shutil.rmtree(self.train_dir)
        shutil.rmtree(self.val_dir)
        self.train_dir = self.val_dir = None

    def test1(self):
        f = io.StringIO()

        with contextlib.redirect_stdout(f):
            try:
                main(['-m', '1', '-f', '-a', AUDIO_DIR, '-i', self.train_dir])
            except SystemExit:
                pass

        min_epc = MakeImagesConfig.min_examples_per_class
        expected_files = [f'XC503001.ogg-{k}-0.png' for k in range(min_epc)]

        for dir_ in self.train_dir, self.val_dir:
            self.assertEqual(os.listdir(dir_), ['helgui'])
            self.assertEqual(set(os.listdir(join(dir_, 'helgui'))),
                             set(expected_files))
