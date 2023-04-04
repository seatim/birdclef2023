
import librosa
import numpy as np

DEFAULT_N_MELS = 224
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = DEFAULT_N_FFT // 2


def image_from_audio(path, n_mels=DEFAULT_N_MELS, n_fft=DEFAULT_N_FFT,
                     hop_length=DEFAULT_HOP_LENGTH, assert_sr=None):
    """Generate mel spectrogram of audio file.
    """
    audio, sr = librosa.load(path, sr=None)

    if assert_sr:
        assert sr == assert_sr, (path, sr)

    M = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    M *= (1 / np.max(M))
    M += 1e-9
    M = np.log(M)
    M -= np.min(M)
    M *= (1 / np.max(M))
    return M
