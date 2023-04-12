
import os
import re
import shutil
import sys

from os.path import join

import click
import numpy as np

from PIL import Image

from adak.config import TrainConfig
from adak.transform import images_from_audio, EmptyImage


def save_image(img, path):
    img = np.flip(img, axis=0)
    Image.fromarray(np.uint8(255*img), 'L').save(path, 'PNG')


def make_images_for_class(label, cfg):
    cls_dir = join(cfg.audio_dir, label)
    img_count = 0
    img_paths = []

    for name in os.listdir(cls_dir):
        path = join(cls_dir, name)
        try:
            images = images_from_audio(path, cfg)
        except EmptyImage:
            print(f'I: empty image, skipping: {path}')
            continue
        assert len(images), (label, name)

        if cfg.max_examples:
            images = images[:cfg.max_examples - img_count]

        os.makedirs(join(cfg.images_dir, label), exist_ok=True)
        for k, image in enumerate(images):
            img_path = join(cfg.images_dir, label, f'{name}-{k}.png')
            save_image(image, img_path)
            img_paths.append(img_path)

        img_count += len(images)
        if cfg.max_examples and img_count >= cfg.max_examples:
            break

    copy_count = 0
    while img_count < cfg.min_examples:
        src_path = img_paths[img_count % len(img_paths)]
        dst_path = re.sub('.png', f'-{copy_count}.png', src_path)
        shutil.copyfile(src_path, dst_path)
        img_count += 1
        copy_count += 1

    assert cfg.min_examples <= img_count <= (cfg.max_examples or img_count), \
        label
    return img_count


@click.command()
@click.option('-a', '--audio-dir', default=TrainConfig.audio_dir,
              show_default=True)
@click.option('-i', '--images-dir', default=TrainConfig.images_dir,
              show_default=True)
@click.option('-m', '--min-examples-per-class', 'min_examples',
              default=TrainConfig.min_examples, show_default=True)
@click.option('-M', '--max-examples-per-class', 'max_examples', type=int)
@click.option('-d', '--frame-duration', default=TrainConfig.frame_duration,
              show_default=True)
def main(audio_dir, images_dir, min_examples, max_examples, frame_duration):
    if min_examples < 1:
        sys.exit('E: min_examples must be > 0')

    if max_examples is not None:
        if max_examples < 1:
            sys.exit('E: max_examples must be > 0')
        if max_examples < min_examples:
            sys.exit('E: max_examples must be >= min_examples')

    config = TrainConfig.from_dict(
        audio_dir=audio_dir, images_dir=images_dir, min_examples=min_examples,
        max_examples=max_examples, frame_duration=frame_duration)

    class_counts = {
        cls: make_images_for_class(cls, config)
        for cls in os.listdir(audio_dir)
    }

    values = class_counts.values()
    print(f'class count stats (min/mean/max):', min(values), '/',
          '%.1f' % np.mean(list(values)), '/', max(values))


if __name__ == '__main__':
    main()
