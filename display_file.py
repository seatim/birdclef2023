
import sys

from os.path import basename, join

import click
import librosa
import matplotlib.pyplot as plt
import numpy as np

MIN_N_FFT = 128
MIN_N_MELS = 128
MAX_N_MELS = 232
DEFAULT_OUTPUT_DIR = '.'


def show_or_save(show, save, array, dir_, filename):
    if show:
        plt.imshow(array, interpolation='nearest', origin='lower')
        plt.show()
    if save:
        plt.imsave(join(dir_, filename), array, origin='lower')


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

    print('Num. samples / FFT frame:', n_fft)
    print('Frame duration:', n_fft / sr)
    print('Num. frames:', len(audio) / n_fft)
    D = librosa.stft(audio, n_fft=n_fft)
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

    M = librosa.feature.melspectrogram(S=np.abs(D), sr=sr, n_mels=n_mels)
    # print('M.shape', M.shape)
    assert np.min(M) >= 0, np.min(M)

    if show_mel_spectrogram or save_mel_spectrogram:
        mel = M * (1 / np.max(M))
        mel += 1e-9
        mel = np.log(mel)

        show_or_save(show_mel_spectrogram, save_mel_spectrogram, mel,
                     output_dir, f'{basename(path)}.mel{n_fft}_{n_mels}.png')


if __name__ == '__main__':
    main()
