
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

from fastai.vision.all import (vision_learner, error_rate, ImageDataLoaders,
                               RandomSplitter, DataBlock, ImageBlock,
                               CategoryBlock, PILImageBW, get_image_files,
                               parent_label, Resize, Brightness, Contrast)
from sklearn.metrics import average_precision_score
from torch.nn.functional import one_hot
from PIL import Image, UnidentifiedImageError

from adak.config import TrainConfig
from adak.sed import SoundEventDetectionFilter, bind_alt


def avg_precision(y_pred, y_true, n_classes):
    assert y_pred.shape[1] == n_classes, y_pred.shape
    return average_precision_score(one_hot(y_true, n_classes), y_pred)


def get_image_info(path):
    f = Image.open(path)
    f.close()
    return f


def get_image_data(path):
    f = Image.open(path)
    data = list(f.getdata())
    f.close()
    return data


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


def check_images(cfg, classes, check_load_images, exit_on_error):
    class_counts = defaultdict(int)

    for label in classes:
        for name in os.listdir(join(cfg.images_dir, label)):
            img_path = join(cfg.images_dir, label, name)
            error = check_image(cfg, img_path, check_load_images)
            if error:
                if exit_on_error:
                    sys.exit(f'E: {error}')
                else:
                    print(f'W: {error}')
            class_counts[label] += 1

    return class_counts


def get_data_loader(path, vocab, cfg, sed, img_cls=PILImageBW):

    splitter = RandomSplitter(cfg.valid_pct, seed=cfg.random_seed)
    item_tfms = Resize(cfg.n_mels)
    batch_tfms = [Brightness(), Contrast()]
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
@click.option('-e', '--epochs', default=5, show_default=True)
def main(check_load_images, exit_on_error, images_dir, epochs):
    if not isdir(images_dir):
        sys.exit(f'E: no such directory: {images_dir}\n\nYou can create an '
                 f'images directory with make_images_from_audio.py.')

    config = TrainConfig.from_dict(images_dir=images_dir)

    tmd = pd.read_csv(join(images_dir, '..', 'train_metadata.csv'))
    classes = np.unique(tmd.primary_label)

    class_counts = check_images(
        config, classes, check_load_images, exit_on_error)
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

    dls = get_data_loader(images_dir, classes, config, sed)
    arch = 'efficientnet_b0'

    metrics = [error_rate]
    if sys.version_info[:2] >= (3, 8):
        # average_precision_score() requires python 3.8+ and version 1.1+ of
        # scikit-learn.  See [1] for more information.
        # [1] https://github.com/scikit-learn/scikit-learn/pull/19085
        metrics.append(partial(avg_precision, n_classes=len(classes)))

    learn = vision_learner(dls, arch, metrics=metrics, cbs=cbs).to_fp16()

    warnings.filterwarnings(
        action='ignore', category=UserWarning,
        message='No positive class found in y_true, recall')

    # TODO: optimize learning rate
    learn.fine_tune(epochs, 0.01)

    timestamp = datetime.now().strftime('%Y%m%d.%H%M%S')
    model_filename = f'birdclef-model-{timestamp}.pkl'
    learn.export(model_filename)
    print(f'exported model to "{model_filename}"')


if __name__ == '__main__':
    main()
