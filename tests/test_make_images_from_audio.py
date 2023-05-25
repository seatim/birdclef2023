
import contextlib
import io
import os
import re
import tempfile
import unittest

from os.path import dirname, join

from adak.config import MakeImagesConfig
from make_images_from_audio import main

from .test_transform import MaskWarnings


class Test_make_images_from_audio(MaskWarnings):
    def setUp(self):
        super().setUp()

        self.train_dir = tempfile.mkdtemp('train')
        self.val_dir = re.sub('train', 'val', self.train_dir)
        os.mkdir(self.val_dir)

    def tearDown(self):
        super().tearDown()

        for dir_ in self.train_dir, self.val_dir:
            for label in os.listdir(dir_):
                label_dir = join(dir_, label)

                for name in os.listdir(label_dir):
                    os.remove(join(label_dir, name))

                os.rmdir(label_dir)

            os.rmdir(dir_)
        self.train_dir = self.val_dir = None

    def test1(self):
        f = io.StringIO()

        with contextlib.redirect_stdout(f):
            try:
                main(['-m', '1', '-f',
                      '-a', dirname(__file__), '-i', self.train_dir])
            except SystemExit:
                pass

        min_epc = MakeImagesConfig.min_examples_per_class
        expected_files = [f'XC503001.ogg-{k}-0.png' for k in range(min_epc)]

        for dir_ in self.train_dir, self.val_dir:
            self.assertEqual(os.listdir(dir_), ['data'])
            self.assertEqual(set(os.listdir(join(dir_, 'data'))),
                             set(expected_files))
