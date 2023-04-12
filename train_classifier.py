
import os
import warnings

from collections import defaultdict
from datetime import datetime
from functools import partial
from os.path import join

import click
import numpy as np
import pandas as pd

from fastai.vision.all import (vision_learner, error_rate, ImageDataLoaders,
                               RandomSplitter, DataBlock, ImageBlock,
                               CategoryBlock, PILImageBW,
                               get_image_files, parent_label, Resize,
                               Brightness, Contrast)
from sklearn.metrics import average_precision_score
from torch.nn.functional import one_hot
from PIL import Image, UnidentifiedImageError

from adak.transform import DEFAULT_N_MELS as N_MELS

DEFAULT_IMAGES_DIR = 'data/train_images'

# for reproducibility
RANDOM_SEED = 11462  # output of random.randint(0, 99999)


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


def check_image(image_path, check_load_image):
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

    if img_shape[1] != N_MELS:
        return f'image height != {N_MELS}: {image_path}'


def check_images(images_dir, classes, check_load_images):
    class_counts = defaultdict(int)

    for label in classes:
        for name in os.listdir(join(images_dir, label)):
            check_image(join(images_dir, label, name), check_load_images)
            class_counts[label] += 1

    return class_counts


def get_data_loader(path, vocab, valid_pct=0.2, seed=RANDOM_SEED,
                    img_cls=PILImageBW):
    splitter = RandomSplitter(valid_pct, seed=seed)
    item_tfms = Resize(N_MELS)
    batch_tfms = [Brightness(), Contrast()]
    dblock = DataBlock(blocks=(ImageBlock(img_cls), CategoryBlock(vocab=vocab)),
                       get_items=get_image_files,
                       splitter=splitter,
                       get_y=parent_label,
                       item_tfms=item_tfms,
                       batch_tfms=batch_tfms)
    return ImageDataLoaders.from_dblock(dblock, path, path=path)


@click.command()
@click.option('-c', '--check-load-images', is_flag=True)
@click.option('-i', '--images-dir', default=DEFAULT_IMAGES_DIR,
              show_default=True)
@click.option('-e', '--epochs', default=5, show_default=True)
def main(check_load_images, images_dir, epochs):
    tmd = pd.read_csv(join(images_dir, '..', 'train_metadata.csv'))
    classes = np.unique(tmd.primary_label)

    class_counts = check_images(images_dir, classes, check_load_images)
    values = class_counts.values()
    print(f'I: class count stats (min/mean/max):', min(values), '/',
          '%.1f' % np.mean(list(values)), '/', max(values))
    print(f'I: training on {sum(values)} image files')

    dls = get_data_loader(images_dir, classes)
    arch = 'efficientnet_b0'

    metrics = [error_rate, partial(avg_precision, n_classes=len(classes))]
    learn = vision_learner(dls, arch, metrics=metrics).to_fp16()

    warnings.filterwarnings(
        action='ignore', category=UserWarning,
        message='No positive class found in y_true, recall')

    # TODO: optimize learning rate
    learn.fine_tune(epochs, 0.01)

    timestamp = datetime.now().strftime('%Y%m%d.%H%M%S')
    model_filename = f'birdclef-model-{timestamp}.pkl'
    learn.save(model_filename)
    learn.export(model_filename)
    print(f'saved and exported model to "{model_filename}"')


if __name__ == '__main__':
    main()
