
import os
import sys

from os.path import join

import click
import numpy as np

from PIL import Image

from adak.transform import images_from_audio, DEFAULT_SAMPLE_RATE

AUDIO_DIR = 'data/train_audio'
DEFAULT_IMAGES_DIR = 'data/train_images'
MAX_PLAY_TIME = 10.


def save_image(img, path):
    img = np.flip(img, axis=0)
    Image.fromarray(np.uint8(255*img), 'L').save(path, 'PNG')


def make_images_for_class(label, images_dir, max_examples):
    cls_dir = join(AUDIO_DIR, label)
    img_count = 0

    for name in os.listdir(cls_dir):
        images = images_from_audio(
            join(cls_dir, name), assert_sr=DEFAULT_SAMPLE_RATE)
        assert len(images), (label, name)

        if max_examples:
            images = images[:max_examples - img_count]

        os.makedirs(join(images_dir, label), exist_ok=True)
        for k, image in enumerate(images):
            save_image(image, join(images_dir, label, f'{name}-{k}.png'))

        img_count += len(images)
        if max_examples and img_count >= max_examples:
            break

    assert 0 < img_count <= (max_examples or img_count), label
    return img_count


@click.command()
@click.option('-i', '--images-dir', default=DEFAULT_IMAGES_DIR,
              show_default=True)
@click.option('-m', '--max-examples-per-class', 'max_examples', type=int)
def main(images_dir, max_examples):
    if max_examples is not None and max_examples < 1:
        sys.exit('E: max_examples must be > 0')

    class_counts = {cls: make_images_for_class(cls, images_dir, max_examples)
                    for cls in os.listdir(AUDIO_DIR)}

    values = class_counts.values()
    print(f'class count stats (min/mean/max):', min(values), '/',
          '%.1f' % np.mean(list(values)), '/', max(values))


if __name__ == '__main__':
    main()
