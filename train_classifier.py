
import os
import sys
import warnings

from datetime import datetime
from functools import partial
from os.path import abspath, isdir, join

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastai.metrics import AccumMetric, ActivationType
from fastai.vision.all import (vision_learner, error_rate, ImageDataLoaders,
                               RandomSplitter, DataBlock, ImageBlock,
                               CategoryBlock, PILImageBW, get_image_files,
                               parent_label, Resize, Brightness, Contrast)

from adak.check import check_images
from adak.config import TrainConfig
from adak.glue import avg_precision, StratifiedSplitter
from adak.pretrain import make_pretrain_learner


def handle_missing_classes(classes, classes_present, prune_missing_classes):
    missing = classes - classes_present
    if missing:
        if prune_missing_classes:
            classes -= missing
            print(f'I: removed {len(missing)} missing classes')
        else:
            print(f'W: found no examples of {len(missing)} classes.  Consider '
                  f'using the --prune-missing-classes option to remove them.')


def create_combined_images_dir(cfg, dry_run=False):
    def dry_run_cmd(cmd):
        print(f'[DRY RUN] {cmd}')

    path = cfg.combined_images_dir
    cmd = dry_run_cmd if dry_run else os.system
    cmd(f'mkdir -p {path}')
    print()
    print('Removing all images from combined images directory...')
    cmd(f'find {path} -name \*.png -delete')
    print()

    if cfg.bc23_images_dir:
        print('Copying images...')
        cmd(f'cp -r {cfg.bc23_images_dir}/* {path}')

    if cfg.bc21_images_dir:
        print('Copying images...')
        cmd(f'cp -r {cfg.bc21_images_dir}/* {path}')

    if cfg.bc22_images_dir:
        print('Copying images...')
        cmd(f'cp -r {cfg.bc22_images_dir}/* {path}')
    print()


def fine_tune_learner(classes, dls, random_split, config, epochs, cpu,
                      pre_learn=None):
    metrics = [error_rate]
    if sys.version_info[:2] >= (3, 8):
        ap_score = AccumMetric(partial(avg_precision, n_classes=len(classes)),
                               activation=ActivationType.Sigmoid,
                               flatten=False)

        # average_precision_score() requires python 3.8+ and version 1.1+ of
        # scikit-learn.  See [1] for more information.
        # [1] https://github.com/scikit-learn/scikit-learn/pull/19085
        metrics.append(ap_score)

    if pre_learn:
        learn = make_pretrain_learner(pre_learn, dls, metrics)
    else:
        learn = vision_learner(dls, config.arch, metrics=metrics)

    if not cpu:
        learn = learn.to_fp16()

    if random_split:
        print('W: average precision score is invalid due to random split')
        warnings.filterwarnings(
            action='ignore', category=UserWarning,
            message='No positive class found in y_true, recall')

    learn.fine_tune(epochs, config.learn_rate)

    return learn


def get_data_loader(vocab, cfg, random_split, show_batch=False,
                    img_cls=PILImageBW):

    path = cfg.combined_images_dir
    splitter_cls = RandomSplitter if random_split else StratifiedSplitter
    splitter = splitter_cls(cfg.valid_pct, cfg.random_seed)
    item_tfms = Resize(cfg.n_mels)
    batch_tfms = [Brightness(0.8), Contrast(0.8)]

    dblock = DataBlock(blocks=(ImageBlock(img_cls), CategoryBlock(vocab=vocab)),
                       get_items=get_image_files,
                       splitter=splitter,
                       get_y=parent_label,
                       item_tfms=item_tfms,
                       batch_tfms=batch_tfms)

    dls = ImageDataLoaders.from_dblock(dblock, path, path=path)
    if show_batch:
        dls.show_batch()
        plt.show()
    return dls


def pretrain_classifier(combined_images_dir, bc21_images_dir, bc22_images_dir,
                        epochs, cpu, check_load_images, exit_on_error,
                        prune_missing_classes, show_batch):

    config = TrainConfig.from_dict(
        bc23_images_dir=None,
        bc21_images_dir=bc21_images_dir, bc22_images_dir=bc22_images_dir,
        combined_images_dir=combined_images_dir)

    classes = set()

    if bc21_images_dir:
        tmd21 = pd.read_csv(join(bc21_images_dir, '..', 'train_metadata.csv'))
        classes |= set(tmd21.primary_label)

    if bc22_images_dir:
        tmd22 = pd.read_csv(join(bc22_images_dir, '..', 'train_metadata.csv'))
        classes |= set(tmd22.primary_label)

    create_combined_images_dir(config)
    classes_present = check_images(config, check_load_images, exit_on_error)
    handle_missing_classes(classes, classes_present, prune_missing_classes)

    dls = get_data_loader(classes, config, False, show_batch)

    return fine_tune_learner(classes, dls, False, config, epochs, cpu)


@click.command()
@click.option('-c', '--check-load-images', is_flag=True)
@click.option('-b', '--exit-on-error', is_flag=True)
@click.option('-i', '--bc23-images-dir', default=TrainConfig.bc23_images_dir,
              show_default=True)
@click.option('-B', '--bc21-images-dir', default=TrainConfig.bc21_images_dir,
              show_default=True)
@click.option('-D', '--bc22-images-dir', default=TrainConfig.bc22_images_dir,
              show_default=True)
@click.option('-I', '--combined-images-dir',
              default=TrainConfig.combined_images_dir, show_default=True)
@click.option('-e', '--epochs', default=TrainConfig.n_epochs, show_default=True)
@click.option('-C', '--cpu', is_flag=True)
@click.option('-r', '--random-split', is_flag=True)
@click.option('-p', '--prune-missing-classes', is_flag=True)
@click.option('-w', '--show-batch', is_flag=True)
def main(check_load_images, exit_on_error, bc23_images_dir, bc21_images_dir,
         bc22_images_dir, combined_images_dir, epochs, cpu, random_split,
         prune_missing_classes, show_batch):

    if not isdir(bc23_images_dir):
        sys.exit(f'E: no such directory: {bc23_images_dir}\n\nYou can create '
                 f'an images directory with make_images_from_audio.py.')

    for dir_, name in ((bc23_images_dir, 'bc23_images_dir'),
                       (bc21_images_dir, 'bc21_images_dir'),
                       (bc22_images_dir, 'bc22_images_dir')):
        if dir_ and abspath(dir_) == abspath(combined_images_dir):
            sys.exit(f'E: {name} and combined_images_dir must be different')

    if bc21_images_dir or bc22_images_dir:
        print('Pretraining model on bc21 and/or bc22 data sets')
        pre_learn = pretrain_classifier(
            combined_images_dir, bc21_images_dir, bc22_images_dir, epochs, cpu,
            check_load_images, exit_on_error, prune_missing_classes,
            show_batch)
    else:
        pre_learn = None

    config = TrainConfig.from_dict(
        bc23_images_dir=bc23_images_dir,
        bc21_images_dir=None, bc22_images_dir=None,
        combined_images_dir=combined_images_dir)

    tmd = pd.read_csv(join(bc23_images_dir, '..', 'train_metadata.csv'))
    classes = set(tmd.primary_label)

    create_combined_images_dir(config)
    classes_present = check_images(config, check_load_images, exit_on_error)
    handle_missing_classes(classes, classes_present, prune_missing_classes)

    dls = get_data_loader(classes, config, random_split, show_batch)

    learn = fine_tune_learner(
        classes, dls, random_split, config, epochs, cpu, pre_learn)

    timestamp = datetime.now().strftime('%Y%m%d.%H%M%S')
    model_path = f'birdclef-model-{timestamp}.pkl'
    learn.export(model_path)
    print(f'exported model to "{model_path}"')


if __name__ == '__main__':
    main()
