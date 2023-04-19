
import os
import sys
import warnings

from collections import defaultdict
from datetime import datetime
from functools import partial
from os.path import isdir, join

import click
import numpy as np
import pandas as pd

from fastai.metrics import AccumMetric, ActivationType
from fastai.vision.all import (vision_learner, error_rate, ImageDataLoaders,
                               RandomSplitter, DataBlock, ImageBlock,
                               CategoryBlock, PILImageBW, get_image_files,
                               parent_label, Resize, Brightness, Contrast)
from PIL import Image, UnidentifiedImageError

from adak.config import TrainConfig
from adak.glue import avg_precision, StratifiedSplitter
from adak.sed import SoundEventDetectionFilter, bind_alt

DEFAULT_COMBINED_IMAGES_DIR = 'data/train_images'


def get_image_info(path):
    f = Image.open(path)
    f.close()
    return f


def get_image_data(path):
    f = Image.open(path)
    data = list(f.getdata())
    f.close()
    return data


def create_combined_images_dir(cfg, path, dry_run=False):
    def dry_run_cmd(cmd):
        print(f'[DRY RUN] {cmd}')

    cmd = dry_run_cmd if dry_run else os.system
    cmd(f'mkdir -p {path}')
    print()
    print('Removing existing files from combined images directory...')
    cmd(f'find {path} -type f -delete')
    print()
    print('Copying images...')
    cmd(f'cp -r {cfg.images_dir}/* {path}')
    if cfg.bc21_images_dir:
        cmd(f'cp -r {cfg.bc21_images_dir}/* {path}')
    print()


def check_image(cfg, image_path, check_load_image):
    try:
        img_shape = get_image_info(image_path).size
    except UnidentifiedImageError as e:
        return e.args[0]

    if check_load_image:
        try:
            expected_image_len = img_shape[0] * img_shape[1]
            assert len(get_image_data(image_path)) == expected_image_len
        except OSError as e:
            return f'{e.args[0]}: {image_path}'

    if img_shape[1] != cfg.n_mels:
        return f'image height != {cfg.n_mels}: {image_path}'


def check_images(cfg, check_load_images, exit_on_error, combined_images_dir):
    class_counts = defaultdict(int)
    classes_present = set(name for name in os.listdir(combined_images_dir)
                          if isdir(join(combined_images_dir, name)))
    classes_present -= {'models'}
    print(f'Found {len(classes_present)} classes in {combined_images_dir}')

    for k, label in enumerate(classes_present):
        print(f'Checking {label} [{k}/{len(classes_present)}]...',
              end='\r', flush=True)

        for name in os.listdir(join(combined_images_dir, label)):
            img_path = join(combined_images_dir, label, name)
            error = check_image(cfg, img_path, check_load_images)
            if error:
                if exit_on_error:
                    sys.exit(f'E: {error}')
                else:
                    print(f'W: {error}')
            class_counts[label] += 1

    return class_counts


def get_data_loader(path, vocab, cfg, sed, random_split, img_cls=PILImageBW):

    splitter_cls = RandomSplitter if random_split else StratifiedSplitter
    splitter = splitter_cls(cfg.valid_pct, cfg.random_seed)
    item_tfms = Resize(cfg.n_mels)
    batch_tfms = [Brightness(0.8), Contrast(0.8)]
    get_y = bind_alt(sed.get_y) if sed else parent_label

    dblock = DataBlock(blocks=(ImageBlock(img_cls), CategoryBlock(vocab=vocab)),
                       get_items=get_image_files,
                       splitter=splitter,
                       get_y=get_y,
                       item_tfms=item_tfms,
                       batch_tfms=batch_tfms)

    return ImageDataLoaders.from_dblock(dblock, path, path=path)


@click.command()
@click.option('-c', '--check-load-images', is_flag=True)
@click.option('-b', '--exit-on-error', is_flag=True)
@click.option('-i', '--images-dir', default=TrainConfig.images_dir,
              show_default=True)
@click.option('-B', '--bc21-images-dir', default=TrainConfig.bc21_images_dir,
              show_default=True)
@click.option('-I', '--combined-images-dir',
              default=DEFAULT_COMBINED_IMAGES_DIR, show_default=True)
@click.option('-e', '--epochs', default=5, show_default=True)
@click.option('-C', '--cpu', is_flag=True)
@click.option('-r', '--random-split', is_flag=True)
def main(check_load_images, exit_on_error, images_dir, bc21_images_dir,
         combined_images_dir, epochs, cpu, random_split):

    if not isdir(images_dir):
        sys.exit(f'E: no such directory: {images_dir}\n\nYou can create an '
                 f'images directory with make_images_from_audio.py.')

    config = TrainConfig.from_dict(
        images_dir=images_dir, bc21_images_dir=bc21_images_dir)

    tmd = pd.read_csv(join(images_dir, '..', 'train_metadata.csv'))
    classes = np.unique(tmd.primary_label)

    if bc21_images_dir:
        tmd21 = pd.read_csv(join(bc21_images_dir, '..', 'train_metadata.csv'))
        classes21 = np.unique(tmd21.primary_label)
        classes = set(classes) | set(classes21)

    create_combined_images_dir(config, combined_images_dir)
    class_counts = check_images(
        config, check_load_images, exit_on_error, combined_images_dir)
    values = class_counts.values()
    print(f'I: class count stats (min/mean/max):', min(values), '/',
          '%.1f' % np.mean(list(values)), '/', max(values))
    print(f'I: training on {sum(values)} image files')

    if config.use_sed:
        classes = list(classes) + [SoundEventDetectionFilter.NON_EVENT]
        sed = SoundEventDetectionFilter()
        cbs = [sed]
    else:
        sed = cbs = None

    dls = get_data_loader(
        combined_images_dir, classes, config, sed, random_split)
    dls.show_batch()
    arch = 'efficientnet_b0'

    metrics = [error_rate]
    if sys.version_info[:2] >= (3, 8):
        ap_score = AccumMetric(partial(avg_precision, n_classes=len(classes)),
                               activation=ActivationType.Sigmoid,
                               flatten=False)

        # average_precision_score() requires python 3.8+ and version 1.1+ of
        # scikit-learn.  See [1] for more information.
        # [1] https://github.com/scikit-learn/scikit-learn/pull/19085
        metrics.append(ap_score)

    learn = vision_learner(dls, arch, metrics=metrics, cbs=cbs)
    if not cpu:
        learn = learn.to_fp16()

    if random_split:
        print('W: average precision score is invalid due to random split')
        warnings.filterwarnings(
            action='ignore', category=UserWarning,
            message='No positive class found in y_true, recall')

    # TODO: optimize learning rate
    learn.fine_tune(epochs, 0.01)

    timestamp = datetime.now().strftime('%Y%m%d.%H%M%S')
    model_path = f'birdclef-model-{timestamp}.pkl'
    learn.export(model_path)
    print(f'exported model to "{model_path}"')


if __name__ == '__main__':
    main()
