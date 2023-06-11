
import os
import re
import shutil
import tempfile
import unittest

from os.path import join

from adak.config import MakeImagesConfig
from make_images_from_audio import main as make_images_from_audio

from . import run_main
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
        args = ['-m', '1', '-f', '-a', AUDIO_DIR, '-i', self.train_dir]
        output = run_main(make_images_from_audio, args)

        min_epc = MakeImagesConfig.min_examples_per_class
        val_expected_files = {
            'helgui': [f'XC503001.ogg-{k}-0.png' for k in range(min_epc)],
            'subbus1': [f'XC390820.ogg-{k}-0.png' for k in range(min_epc)]
        }
        train_expected_files = dict(val_expected_files)
        train_expected_files['train_metadata.csv'] = None

        for dir_, expected_files in ((self.train_dir, train_expected_files),
                                     (self.val_dir, val_expected_files)):
            self.assertEqual(set(os.listdir(dir_)), set(expected_files.keys()))

            for name, files in expected_files.items():
                if files is not None:
                    self.assertEqual(
                        set(os.listdir(join(dir_, name))), set(files))
