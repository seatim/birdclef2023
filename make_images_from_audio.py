
import os
import re
import sys

from collections import defaultdict
from itertools import chain
from os.path import basename, exists, join, normpath
from pathlib import Path

import click
import numpy as np

from fastai.data.all import get_files, parent_label
from sklearn.utils import resample
from PIL import Image

from adak.config import MakeImagesConfig
from adak.glue import StratifiedSplitter
from adak.transform import images_from_audio, EmptyImage


def save_image(img, path):
    img = np.flip(img, axis=0)
    Image.fromarray(np.uint8(255*img), 'L').save(path, 'PNG')


def ensure_minimum_example_counts(paths, valid_pct):
    minimum = max(2, int(1 / valid_pct))

    paths_by_class = defaultdict(list)
    for path in paths:
        paths_by_class[parent_label(path)].append(path)

    result = []
    n_copies = 0

    for label, paths in paths_by_class.items():
        if len(paths) < minimum:
            n_copies += minimum - len(paths)
            paths = resample(paths, n_samples=minimum)

        result.extend(paths)

    return result, n_copies


def train_val_split(paths, cfg):
    # split deterministically because we only need to do this once.
    splitter = StratifiedSplitter(cfg.valid_pct, 0)
    paths, n_copies = ensure_minimum_example_counts(paths, cfg.valid_pct)
    print(f'Copying {n_copies} files to ensure minimum example counts for '
          f'train/val split')
    print()
    print('Split stats:')
    train, val = splitter(paths)
    print()
    paths = np.array(paths)
    return paths[train], paths[val]


def sample_image_addresses(images_by_path, n_images_total, n_samples, label,
                           config):
    paths = set(images_by_path.keys())

    def addresses_for_path(path):
        return ((path, j) for j in range(len(images_by_path[path])))

    all_addresses = list(chain(*(addresses_for_path(p) for p in paths)))

    assert len(all_addresses) == n_images_total, \
        (label, len(all_addresses), n_images_total)

    assert all(len(a) == 2 for a in all_addresses)
    assert all(isinstance(a[0], Path) for a in all_addresses)
    assert all(isinstance(a[1], int) for a in all_addresses)

    assert len(set(all_addresses)) == len(all_addresses), \
        (label, len(set(all_addresses)), len(all_addresses))

    replace = n_images_total < n_samples
    retries = 0

    while retries < config.sample_retries:
        image_addresses = resample(
            all_addresses, n_samples=n_samples, replace=replace)

        # try to sample from all files
        sample_paths = set(a[0] for a in image_addresses)
        unused_files = paths - sample_paths

        if unused_files and n_samples >= len(paths):
            retries += 1
        else:
            break

    if retries == config.sample_retries:
        print(f'W: failed after {config.sample_retries} retries to sample '
              f'from all {len(paths)} files for class {label}')

    return image_addresses


def make_images_for_class(label, paths, images_dir, config, verbose=False):
    """Extract a random sample of possible images from audio files for a class.

    NB: files are different sizes; a different number of images can be
    extracted from each file.
    """
    assert len(set(parent_label(path) for path in paths)) == 1, list(paths)[0]

    if config.max_paths_per_class:
        np.random.shuffle(paths)
        paths = paths[:config.max_paths_per_class]
    # print(f'Using {len(paths)} paths for {label}')

    def path_images(path):
        try:
            return path, images_from_audio(
                path, config, max_frames=config.max_images_per_file)
        except EmptyImage:
            print(f'I: empty image, skipping: {path}')
            return None

    images_by_path = dict(filter(None, (path_images(path) for path in paths)))

    n_images_total = sum(len(imgs) for imgs in images_by_path.values())
    # print(f'Found {n_images_total} images for {label} (mean '
    #       f'{n_images_total / len(paths) : .1f})')
    n_samples = min(config.max_examples_per_class,
                    max(config.min_examples_per_class, n_images_total))

    sample_addresses = sample_image_addresses(
        images_by_path, n_images_total, n_samples, label, config)

    indices_by_path = defaultdict(list)
    for path, index in sample_addresses:
        indices_by_path[path].append(index)

    for path, indices in indices_by_path.items():
        images = images_by_path[path]

        for j, k in enumerate(indices):
            img_path = join(images_dir, label, f'{basename(path)}-{j}-{k}.png')
            save_image(images[k], img_path)

    if verbose:
        print(f'I: created {len(sample_addresses)} images for class {label}   ')
    return len(sample_addresses)


def make_images(split_name, audio_files, images_dir, cfg, verbose=False):
    paths_by_class = defaultdict(list)
    for path in audio_files:
        paths_by_class[parent_label(path)].append(path)

    os.makedirs(images_dir, exist_ok=True)
    os.system(f'find {images_dir} -type f -delete')

    class_counts = {}

    print(f'Making {split_name} images...')
    for k, (label, paths) in enumerate(paths_by_class.items()):
        os.makedirs(join(images_dir, label), exist_ok=True)

        print(f'Making images for {label} [{k}/{len(paths_by_class)}]...',
              end='\r', flush=True)
        n_imgs = make_images_for_class(label, paths, images_dir, cfg, verbose)
        class_counts[label] = n_imgs

    values = class_counts.values()
    print(f'{split_name} class count stats (min/mean/max):', min(values), '/',
          '%.1f' % np.mean(list(values)), '/', max(values))
    print()


@click.command()
@click.option('-a', '--audio-dir', default=MakeImagesConfig.audio_dir,
              show_default=True)
@click.option('-i', '--images-dir', default=MakeImagesConfig.images_dir,
              show_default=True)
@click.option('-m', '--max-images-per-file', type=int)
@click.option('-M', '--max-paths-per-class', type=int)
@click.option('-f', '--force-overwrite', is_flag=True)
@click.option('-S', '--no-split', is_flag=True)
@click.option('-v', '--verbose', is_flag=True)
def main(audio_dir, images_dir, max_images_per_file, max_paths_per_class,
         force_overwrite, no_split, verbose):

    if exists(images_dir) and not force_overwrite:
        sys.exit(f'E: {images_dir} exists.  Use "-f" to overwrite it.')

    if 'train' not in basename(normpath(images_dir)):
        sys.exit(f'E: images_dir must have "train" in its name.')

    config = MakeImagesConfig.from_dict(
        audio_dir=audio_dir, images_dir=images_dir,
        max_images_per_file=max_images_per_file,
        max_paths_per_class=max_paths_per_class)

    assert 0 < config.min_examples_per_class <= config.max_examples_per_class

    audio_files = get_files(audio_dir, '.ogg')
    print(f'Found {len(audio_files)} audio files')

    if no_split:
        make_images('train', audio_files, images_dir, config, verbose)
    else:
        # Reserve a fraction of files for validation.
        train, val = train_val_split(audio_files, config)

        make_images('train', train, images_dir, config, verbose)
        make_images('val', val, re.sub('train', 'val', images_dir), config,
                    verbose)


if __name__ == '__main__':
    main()
