
"""Module for turning audio files into images.
"""

import os

import librosa
import numpy as np
import soundfile

from fastai.vision.all import PILImageBW, TensorCategory
from skimage import exposure
from PIL import Image


class EmptyImage(Exception):
    pass


def image_from_audio(path, cfg, max_width=None):
    """Generate mel spectrogram of audio file.
    """
    audio, sr = soundfile.read(path)
    assert sr == cfg.sample_rate, (path, sr)
    audio = librosa.to_mono(audio)

    if os.getenv('FAKE_IMAGE_FROM_AUDIO'):
        return np.zeros((cfg.n_mels, cfg.image_width(len(audio) / sr)))

    if max_width is not None:
        max_samples = (max_width - 1) * cfg.hop_length
        audio = audio[:max_samples]

    M = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=cfg.n_mels, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length)

    if not np.max(M):
        raise EmptyImage

    M *= (1 / np.max(M))
    M += 1e-9
    M = np.log(M)
    M -= np.min(M)
    M *= (1 / np.max(M))
    return M


def images_from_audio(path, cfg, max_frames=None):
    if max_frames:
        max_width = cfg.frame_width + max_frames * cfg.frame_hop_length
    else:
        max_width = None

    image = image_from_audio(path, cfg, max_width)
    remainder = image.shape[1] % cfg.frame_hop_length

    if cfg.frame_hop_length > cfg.frame_width:
        raise ValueError('frame_hop_length must be <= frame_width')

    if remainder and cfg.pad_remainder:
        pad_len = cfg.frame_hop_length - remainder
        image = np.hstack([image, np.zeros((cfg.n_mels, pad_len))])
        assert image.shape[1] % cfg.frame_hop_length == 0
    else:
        # NB: remainder is dropped in this case
        pass

    def frame(k):
        offset = k * cfg.frame_hop_length
        end = offset + cfg.frame_width

        if end <= image.shape[1]:
            return image[:, offset:end]
        else:
            assert cfg.frame_hop_length != cfg.frame_width, \
                'if frame_hop_length == frame_width this should not happen'

            if not cfg.pad_remainder:
                raise ValueError(
                    'pad_remainder is required when frame_hop_length != '
                    'frame_width')

            pad_len = end - image.shape[1]
            return np.hstack(
                [image[:, offset:], np.zeros((cfg.n_mels, pad_len))])

    n_frames = image.shape[1] // cfg.frame_hop_length
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    return [frame(k) for k in range(0, n_frames)]


def center_median(x):
    assert 0 <= np.min(x) <= 1, np.min(x)
    assert 0 <= np.max(x) <= 1, np.max(x)
    median = np.median(x)
    return np.where(x < median, x*(0.5/median), 0.5+(x-median)*(0.5/(1-median)))


def clip_tails(x, n_std=3):
    assert 0 <= np.min(x) <= 1, np.min(x)
    assert 0 <= np.max(x) <= 1, np.max(x)
    median = np.median(x)
    std = np.std(x)
    lo = median - n_std*std
    hi = median + n_std*std
    assert lo <= hi, (median, std, n_std)
    x = np.where(x < lo, lo, np.where(x > hi, hi, x))
    x -= np.min(x)
    x *= (1/np.max(x))
    return x


def add_histeq(img):
    if type(img) is not PILImageBW:
        if type(img) is TensorCategory:
            return img
        assert type(img) is np.ndarray, type(img)
        assert img.dtype == np.uint8, img.dtype
        assert len(img.shape) == 2, img.shape

    array = np.array(img) / 255
    clipped = clip_tails(array)
    p2, p98 = np.percentile(array, (2, 98))
    rescaled = exposure.rescale_intensity(array, in_range=(p2, p98))
    stack = sum([array, clipped, rescaled]) / 3
    return Image.fromarray((stack * 255).astype(np.uint8), 'L')
