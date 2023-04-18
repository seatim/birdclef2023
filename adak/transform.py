
import librosa
import numpy as np
import soundfile


class EmptyImage(Exception):
    pass


def image_from_audio(path, cfg, max_width=None):
    """Generate mel spectrogram of audio file.
    """
    audio, sr = soundfile.read(path)
    assert sr == cfg.sample_rate, (path, sr)

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


def image_width(audio_play_time, sr, hop_length):
    return 1 + int(audio_play_time * sr) // hop_length


def images_from_audio(path, cfg, max_frames=None):
    frame_width = image_width(
        cfg.frame_duration, cfg.sample_rate, cfg.hop_length)
    image = image_from_audio(path, cfg, frame_width if max_frames else None)
    remainder = image.shape[1] % cfg.frame_hop_length

    if cfg.frame_hop_length > frame_width:
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
        end = offset + frame_width

        if end <= image.shape[1]:
            return image[:, offset:end]
        else:
            assert cfg.frame_hop_length != frame_width, \
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
