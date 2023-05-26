
"""Make an NSE file for an image collection **that does not exist on disk**.

This is done by reproducing exactly the images that would be created for a set
of audio files by make_images_from_audio.py, and using SHA1 hashes to identify
those images.

NSE files can be used with make_nse_dir.py and train_classifier.py.
"""

import hashlib
import os
import sys

from os.path import exists
from io import BytesIO

import click
import numpy as np
import pandas as pd
import soundfile

from fastai.data.all import get_files
from PIL import Image

from adak.config import BaseConfig
from adak.sed import sound_event_proba
from adak.transform import images_from_audio


def image_sha1(img):
    buf = BytesIO()
    img = np.flip(img, axis=0)
    Image.fromarray(np.uint8(255*img), 'L').save(buf, 'PNG')
    state = hashlib.sha1()
    state.update(buf.getvalue())
    return state.hexdigest()


@click.command(help=__doc__)
@click.option('-a', '--audio-dir', default=BaseConfig.audio_dir,
              show_default=True)
@click.option('-o', '--outpath', default='nsedata.csv', show_default=True)
def main(audio_dir, outpath):

    if exists(outpath):
        sys.exit(f'E: path exists: {outpath}')

    config = BaseConfig()
    table = []

    f = open(outpath + '.tmp', 'w')  # temp file

    paths = [str(path) for path in get_files(audio_dir, '.ogg')]
    for j, path in enumerate(paths):
        short_name = os.sep.join(path.split(os.sep)[-2:])
        print(f'Reading {short_name} [{j}/{len(paths)}]...', end='\r',
              flush=True)

        images = images_from_audio(path, config)
        audio, sr = soundfile.read(path)
        assert len(audio.shape) == 1, (path, audio.shape)
        assert sr == BaseConfig.sample_rate, (path, sr)

        probs = sound_event_proba(audio, config)
        if len(probs) not in (len(images), len(images) - 1):
            print('len(images) - len(probs) =', len(images) - len(probs), path)

        if len(probs) != len(images):
            assert len(probs) < len(images)
            probs += [0] * (len(images) - len(probs))
            assert len(probs) == len(images)

        sha1s = [image_sha1(img) for img in images]
        img_paths = [f'{path}-{k}.png' for k in range(len(images))]
        rows = list(zip(probs, img_paths, sha1s))
        table.extend(rows)

        # write rows to temp file also, to avoid losing precious data in the
        # event this fine program should succumb to fatigue unexpectedly.
        for row in rows:
            print(','.join([str(x) for x in row]), file=f)

    # "max_bc23" is misleading but keep it for compatibility.  ideally the
    # column would now be renamed "sound_event_proba" or similar.
    df = pd.DataFrame(table, columns=('max_bc23', 'path', 'sha1'))
    df.to_csv(outpath, index=False)

    # if we get to here it is safe to delete the temp file.
    f.close()
    os.remove(f.name)


if __name__ == '__main__':
    main()
