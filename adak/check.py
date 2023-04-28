
import os
import sys

from collections import defaultdict
from os.path import isdir, join

import numpy as np

from PIL import Image, UnidentifiedImageError


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


def check_images(cfg, check_load_images, exit_on_error):
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
            error = check_image(cfg, img_path, check_load_images)
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
