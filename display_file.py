
import re
import sys

from os.path import basename, join

import click
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage import exposure
from tabulate import tabulate

MIN_N_FFT = 128
MIN_N_MELS = 128
MAX_N_MELS = 232
HOP_FACTOR = 2
DEFAULT_OUTPUT_DIR = '.'


def show_or_save(show, save, array, dir_, filename):
    if show:
        plt.imshow(array, interpolation='nearest', origin='lower')
        plt.show()
    if save:
        plt.imsave(join(dir_, filename), array, origin='lower')


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


def histeq(array, dir_, filename, do_show_hist=False):
    print()
    print('histogram equalization')
    print()

    stats = []

    def showhist(array, label):
        values = pd.Series(array.flatten())
        if not (0 <= min(values) <= 1):
            print(f'W: min for {label} is {min(values)}')
        if not (0 <= max(values) <= 1):
            print(f'W: max for {label} is {max(values)}')

        quantiles = np.quantile(values, (0.25, 0.5, 0.75))
        stats.append((label, min(values), *quantiles, max(values),
                      np.std(values)))
        if do_show_hist:
            values.hist(bins=20)
            plt.show()

    showhist(array, 'original')

    # Median centering
    centmed = center_median(array)
    showhist(centmed, 'center median')
    plt.imsave(join(dir_, re.sub('.png', '.centmed.png', filename)), centmed,
               origin='lower')

    # Tail clipping
    clipped = clip_tails(array)
    showhist(clipped, 'clip tails')
    plt.imsave(join(dir_, re.sub('.png', '.cliptails.png', filename)), clipped,
               origin='lower')

    # Contrast stretching
    p2, p98 = np.percentile(array, (2, 98))
    img_rescale = exposure.rescale_intensity(array, in_range=(p2, p98))
    plt.imsave(join(dir_, re.sub('.png', '.rescale.png', filename)),
               img_rescale, origin='lower')
    showhist(img_rescale, 'rescale')

    # Equalization
    img_eq = exposure.equalize_hist(array)
    plt.imsave(join(dir_, re.sub('.png', '.histeq.png', filename)), img_eq,
               origin='lower')
    showhist(img_eq, 'histeq')

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(array, clip_limit=0.03)
    plt.imsave(join(dir_, re.sub('.png', '.adapteq.png', filename)),
               img_adapteq, origin='lower')
    showhist(img_adapteq, 'adapteq')

    print(tabulate(stats, headers='label/min/25%/50%/75%/max/std'.split('/')))
    print()


@click.command()
@click.argument('path', type=click.Path())
@click.option('-w', '--show-waveform', is_flag=True)
@click.option('-n', '--fft-frame-size', 'n_fft', type=int, default=2048,
              show_default=True)
@click.option('-m', '--n-mels', type=int, default=128, show_default=True)
@click.option('-l', '--limit-audio-length', type=float)
@click.option('-x', '--show-magnitude-spectrogram', is_flag=True)
@click.option('-X', '--save-magnitude-spectrogram', is_flag=True)
@click.option('-y', '--show-phase-spectrogram', is_flag=True)
@click.option('-Y', '--save-phase-spectrogram', is_flag=True)
@click.option('-z', '--show-mel-spectrogram', is_flag=True)
@click.option('-Z', '--save-mel-spectrogram', is_flag=True)
@click.option('-o', '--output-dir', default=DEFAULT_OUTPUT_DIR,
              show_default=True)
def main(path, show_waveform, n_fft, n_mels, limit_audio_length,
         show_magnitude_spectrogram, save_magnitude_spectrogram,
         show_phase_spectrogram, save_phase_spectrogram,
         show_mel_spectrogram, save_mel_spectrogram, output_dir):

    if n_fft < MIN_N_FFT:
        sys.exit(f'E: FFT frame size must be >= {MIN_N_FFT}.')

    if n_fft & (n_fft - 1):
        sys.exit('E: FFT frame size must be a power of two.')

    if not (MIN_N_MELS <= n_mels <= MAX_N_MELS):
        sys.exit(f'E: n_mels must be >= {MIN_N_MELS} and <= {MAX_N_MELS}.')

    if n_mels & 7:
        sys.exit(f'E: n_mels must be a multiple of 8.')

    audio, sr = librosa.load(path, sr=None)
    assert len(audio.shape) == 1, audio.shape
    print('Sample rate:', sr)

    play_time = len(audio) / sr
    print('Playback length:', play_time)

    if limit_audio_length and play_time > limit_audio_length:
        print('I: applying audio length limit')
        audio = audio[:int(limit_audio_length * sr)]

    if show_waveform:
        librosa.display.waveshow(audio, sr=sr)
        plt.show()

    hop_length = n_fft // HOP_FACTOR
    print('Num. samples / FFT frame:', n_fft)
    print('Frame duration (seconds):', n_fft / sr)
    print('Num. frames:', 1 + (len(audio) + hop_length - 1) // hop_length)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    # print('D.shape', D.shape)
    magnitude, phase = librosa.magphase(D)
    assert np.min(magnitude) >= 0, np.min(magnitude)

    if show_magnitude_spectrogram or save_magnitude_spectrogram:
        magnitude *= (1 / np.max(magnitude))
        magnitude = np.log(magnitude)

        show_or_save(show_magnitude_spectrogram, save_magnitude_spectrogram,
                     magnitude, output_dir, f'{basename(path)}.mag{n_fft}.png')

    if show_phase_spectrogram or save_phase_spectrogram:
        phase = np.angle(phase)
        phase -= np.min(phase)
        phase *= (1 / np.max(phase))

        show_or_save(show_phase_spectrogram, save_phase_spectrogram, phase,
                     output_dir, f'{basename(path)}.phase{n_fft}.png')

    M = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    # print('M.shape', M.shape)
    assert np.min(M) >= 0, np.min(M)

    if show_mel_spectrogram or save_mel_spectrogram:
        mel = M * (1 / np.max(M))
        mel += 1e-9
        mel = np.log(mel)
        mel -= np.min(mel)
        mel *= (1 / np.max(mel))

        show_or_save(show_mel_spectrogram, save_mel_spectrogram, mel,
                     output_dir, f'{basename(path)}.mel{n_fft}_{n_mels}.png')
        histeq(mel, output_dir, f'{basename(path)}.mel{n_fft}_{n_mels}.png')


if __name__ == '__main__':
    main()
