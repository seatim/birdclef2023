
"""Functions for checking image files.
"""

import os
import sys

from collections import defaultdict
from os.path import isdir, join

import numpy as np

from PIL import Image, UnidentifiedImageError


def check_image(cfg, image_path):
    try:
        img = Image.open(image_path)
    except UnidentifiedImageError as e:
        return e.args[0]

    try:
        img.verify()
    except OSError as e:
        if e.args[0] == 'Truncated File Read':
            return f'{e.args[0]}: {image_path}'
        else:
            raise
    finally:
        img.close()

    if img.size[1] != cfg.n_mels:
        return f'image height != {cfg.n_mels}: {image_path}'


def check_images(cfg, exit_on_error):
    combined_images_dir = cfg.combined_images_dir
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
            error = check_image(cfg, img_path)
            if error:
                if exit_on_error:
                    sys.exit(f'E: {error}')
                else:
                    print(f'W: {error}')
            class_counts[label] += 1

    values = class_counts.values()
    print(f'I: class count stats (min/mean/max):', min(values), '/',
          '%.1f' % np.mean(list(values)), '/', max(values))
    print(f'I: training on {sum(values)} image files')

    return {name for name, count in class_counts.items() if count}
