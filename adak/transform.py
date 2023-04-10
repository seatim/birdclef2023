
import librosa
import numpy as np
import soundfile

DEFAULT_N_MELS = 224
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = DEFAULT_N_FFT // 2
DEFAULT_SAMPLE_RATE = 32000


def image_from_audio(path, n_mels=DEFAULT_N_MELS, n_fft=DEFAULT_N_FFT,
                     hop_length=DEFAULT_HOP_LENGTH, assert_sr=None):
    """Generate mel spectrogram of audio file.
    """
    audio, sr = soundfile.read(path)

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


def image_width(audio_play_time, sr=DEFAULT_SAMPLE_RATE,
                hop_length=DEFAULT_HOP_LENGTH):
    return 1 + int(audio_play_time * sr) // hop_length


def images_from_audio(path, frame_duration=10., pad_remainder=True, **kwargs):
    image = image_from_audio(path, **kwargs)

    try:
        kwargs['sr'] = kwargs.pop('assert_sr')
    except KeyError:
        pass

    max_image_width = image_width(frame_duration, **kwargs)

    remainder = image.shape[1] % max_image_width
    if remainder and pad_remainder:
        image_height = kwargs.get('n_mels', DEFAULT_N_MELS)
        pad_len = max_image_width - remainder
        image = np.hstack([image, np.zeros((image_height, pad_len))])
        assert image.shape[1] % max_image_width == 0
    else:
        # NB: remainder is dropped in this case
        pass

    n_images = image.shape[1] // max_image_width
    return [image[:, k:k+max_image_width]
            for k in range(0, max_image_width * n_images, max_image_width)]
