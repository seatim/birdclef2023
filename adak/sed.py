
import math
import os
import warnings

from collections import defaultdict
from io import BytesIO
from itertools import chain, combinations

from os.path import basename, exists, join

import numpy as np
import pandas as pd
import soundfile
import webrtcvad

from fastai.data.all import parent_label

from .hashfile import file_sha1

VAD = webrtcvad.Vad(0)


def set_intersections(sets, digits=6):
    return [round(len(a & b) / len(a | b), digits)
            for a, b in combinations(sets, 2)]


class SoundEventDetectionFilter:
    COLUMNS = {'path', 'max_bc23', 'sha1'}

    def __init__(self, *paths, threshold=None):
        self.paths = list(paths)
        self.dfs = [pd.read_csv(p) for p in self.paths]
        self.threshold = threshold

        for p, df in zip(paths, self.dfs):
            assert set(df.columns) == self.COLUMNS, (p, df.columns)

    def examine(self):
        sha1s = [set(df.sha1) for df in self.dfs]
        fn_unique_hashes = [round(len(s) / len(df.sha1), 6)
                            for s, df in zip(sha1s, self.dfs)]
        print(f'I: there are {len(set(chain(*sha1s)))} unique hashes.')
        print(f'I: fraction of unique hashes, by file: {fn_unique_hashes}')

        if len(self.dfs) > 1:
            print(f'I: fraction of intersection of hashes, by file pair: '
                  f'{set_intersections(sha1s)}')

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value
        if value is None:
            self.ps = []
            self.nse = []
            return

        if not (0 <= value < 1):
            raise ValueError('threshold must be between 0 and 1')

        self.ps = [round(np.quantile(df['max_bc23'], value), 6)
                   for df in self.dfs]
        self.nse = [df[df['max_bc23'] < p] for df, p in zip(self.dfs, self.ps)]
        print(f'I: NSE thresholds for quantile {value} (max_bc23): {self.ps}')

        if len(self.dfs) > 1:
            print(f'I: fraction of intersection of hashes, by nse set pair: '
                  f'{set_intersections([set(x) for x in self.nse_hashes])}')

    @property
    def nse_hashes(self):
        return [set(df['sha1']) for df in self.nse]

    @property
    def nse_count(self):
        return [len(df['sha1']) for df in self.nse]

    def __contains__(self, sha1):
        return all([sha1 in hashes for hashes in self.nse_hashes])

    def relabel_files(self, dir_):
        nse_dir = join(dir_, 'NSE')
        os.makedirs(nse_dir, exist_ok=True)

        is_train = 'train' in basename(dir_)
        if not is_train:
            assert 'val' in basename(dir_), basename(dir_)

        # for a training dataset we need at least three examples per class for
        # the split, so keep track of the NSE examples here and do not relabel
        # any example until we know we have more than three.
        nse_hashes = defaultdict(set)
        good_hashes = defaultdict(set)
        nse_paths = defaultdict(list)

        for root, dirs, files in os.walk(dir_):

            for name in files:
                if not name.endswith('.png'):
                    continue

                path = join(root, name)
                label = parent_label(path)
                sha1 = file_sha1(path)

                if sha1 in self:
                    nse_hashes[label].add(sha1)
                    nse_paths[sha1].append(path)
                else:
                    good_hashes[label].add(sha1)

        nse_count = sum(len(s) for s in nse_hashes.values())
        print(f'I: identified {nse_count} NSE example hashes')

        move_count = 0
        save_count = 0

        for label, nse in nse_hashes.items():
            if is_train:
                nse = list(nse)
                np.random.shuffle(nse)

                n_good = len(good_hashes[label])
                while n_good < 3 and nse:
                    sha1 = nse.pop()
                    save_count += len(nse_paths[sha1])
                    n_good += 1

            while nse:
                sha1 = nse.pop()
                for path in nse_paths[sha1]:
                    os.rename(path, join(nse_dir, basename(path)))
                    move_count += 1

                # a hash can be referenced by multiple labels.  so, clear the
                # list after moving files (or else encounter FileNotFoundError)
                nse_paths[sha1] = []

        print(f'I: relabeled {move_count} NSE example files')
        print(f'I: passed on relabeling {save_count} NSE example files')


def sound_event_proba(audio, config):
    """Return sound event probabilities for audio file.
    """
    if type(audio) is not np.ndarray:
        raise ValueError('input must be 1D numpy array of floats')
    if len(audio.shape) != 1:
        raise ValueError('input must be 1D numpy array of floats')
    if audio.dtype != float and audio.dtype != np.float32:
        raise ValueError('input must be 1D numpy array of floats')
    if min(audio) < -1.1 or max(audio) > 1.1:
        warnings.warn(f'audio sample values outside of range (-1.1, 1.1): '
                      f'min = {min(audio)}, max = {max(audio)}')

    sr = config.sample_rate
    fd = config.frame_duration
    fhf = config.frame_hop_factor

    assert sr in (8000, 16000, 32000, 48000), sr
    assert fhf == int(fhf), fhf
    assert fd > 0, fd
    assert fhf > 0, fhf

    # sr * fd must be divisible by fhf for the list comprehension below to work
    assert int(sr * fd) % int(fhf) == 0, (sr, fd, fhf)

    # "vad" is "voice activity detection" ... each vad frame corresponds to a
    # spectogram image.
    hop_length = int(sr * fd) // int(fhf)
    n_vad_frames = math.ceil(len(audio) / hop_length)
    vad_frames = [audio[k * hop_length : (k + int(fhf)) * hop_length]
                  for k in range(n_vad_frames)]

    # "mvf" is "micro vad frame" ... 30 msec long segment of vad frame.
    mvfd = .03  # 30 msec, selected from options 10, 20, 30
    bytes_per_mvf = 2 * int(mvfd * sr)
    mvf_per_vad_frame = int(fd / mvfd)

    def has_sound_event(frame):
        """Estimate the probability that an audio frame contains a sound event.

        To do this, convert the audio to 16-bit PCM and divide it into 30 ms
        segments so we can score it with py-webrtcvad.

        """
        if len(frame) < bytes_per_mvf // 2:
            return 0

        buf = BytesIO()
        soundfile.write(buf, frame, sr, 'PCM_16', format='RAW')

        n_mvf = int(len(buf.getvalue()) / bytes_per_mvf)
        segments = [buf.getvalue()[k * bytes_per_mvf : (k + 1) * bytes_per_mvf]
                    for k in range(n_mvf)]
        assert segments, len(frame)

        return sum(VAD.is_speech(segment, sr)
                   for segment in segments) / mvf_per_vad_frame

    return [has_sound_event(frame) for frame in vad_frames]
