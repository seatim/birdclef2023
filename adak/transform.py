
import librosa
import numpy as np
import soundfile


class EmptyImage(Exception):
    pass


def image_from_audio(path, cfg):
    """Generate mel spectrogram of audio file.
    """
    audio, sr = soundfile.read(path)
    assert sr == cfg.sample_rate, (path, sr)

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


def image_width(audio_play_time, sr, hop_length):
    return 1 + int(audio_play_time * sr) // hop_length


def images_from_audio(path, cfg):
    image = image_from_audio(path, cfg)
    max_image_width = image_width(
        cfg.frame_duration, cfg.sample_rate, cfg.hop_length)
    remainder = image.shape[1] % max_image_width

    if remainder and cfg.pad_remainder:
        pad_len = max_image_width - remainder
        image = np.hstack([image, np.zeros((cfg.n_mels, pad_len))])
        assert image.shape[1] % max_image_width == 0
    else:
        # NB: remainder is dropped in this case
        pass

    n_images = image.shape[1] // max_image_width
    return [image[:, k:k+max_image_width]
            for k in range(0, max_image_width * n_images, max_image_width)]
