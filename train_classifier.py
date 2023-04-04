
import os

from os.path import dirname, exists, join

import click
import librosa
import numpy as np
import pandas as pd
import soundfile

from fastai.vision.all import (vision_learner, error_rate, ImageDataLoaders,
                               RandomSplitter, DataBlock, ImageBlock,
                               CategoryBlock, PILImageBW, get_files,
                               get_image_files, parent_label)
from PIL import Image, UnidentifiedImageError

SAMPLE_RATE = 32000
N_MELS = 224
N_FFT = 1024
HOP_LENGTH = N_FFT // 2
AUDIO_DIR = 'data/train_audio'
DEFAULT_IMAGES_DIR = 'data/train_images'

# for reproducibility
RANDOM_SEED = 11462  # output of random.randint(0, 99999)


def image_from_audio(path):
    audio, sr = librosa.load(path, sr=None)
    assert sr == SAMPLE_RATE, (path, sr)
    M = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    M *= (1 / np.max(M))
    M += 1e-9
    M = np.log(M)
    M -= np.min(M)
    M *= (1 / np.max(M))
    return M


def get_image_info(path):
    f = Image.open(path)
    f.close()
    return f


def get_image_data(path):
    f = Image.open(path)
    data = list(f.getdata())
    f.close()
    return data


def check_image(audio_path, image_path, check_load_image):
    audio_len = soundfile.info(audio_path).frames
    expected_img_width = 1 + audio_len // HOP_LENGTH

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

    if img_shape[0] != expected_img_width:
        return f'image width != {expected_img_width}: {image_path}'


def check_image_cache(audio_dir, image_cache_dir, check_load_images):
    if not exists(image_cache_dir):
        os.mkdir(image_cache_dir)

    count = 0
    for path in get_files(audio_dir, '.ogg'):
        classname, filename = str(path).split('/')[-2:]
        img_path = join(image_cache_dir, classname, filename + '.png')

        os.makedirs(dirname(img_path), exist_ok=True)

        if not exists(img_path):
            print(f'I: rendering image for {path}...')
            img = image_from_audio(str(path))
            img = np.flip(img, axis=0)
            Image.fromarray(np.uint8(255*img), 'L').save(img_path, 'PNG')
            assert exists(img_path), img_path
        else:
            result = check_image(str(path), img_path, check_load_images)
            if result:
                print(f'W: {result}')

        count += 1

    return count


def get_data_loader(path, vocab, valid_pct=0.2, seed=RANDOM_SEED,
                    img_cls=PILImageBW):
    splitter = RandomSplitter(valid_pct, seed=seed)

    # TODO
    item_tfms = None
    batch_tfms = None

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
def main(check_load_images, images_dir):
    tmd = pd.read_csv('data/train_metadata.csv')
    classes = np.unique(tmd.primary_label)

    count = check_image_cache(AUDIO_DIR, images_dir, check_load_images)
    print(f'I: confirmed {count} files in image cache')

    dls = get_data_loader(images_dir, classes)
    arch = 'efficientnet_b0'
    learn = vision_learner(dls, arch, metrics=error_rate).to_fp16()


if __name__ == '__main__':
    main()
