

class TrainConfig:
    audio_dir = 'data/train_audio'
    images_dir = 'data/train_images'
    n_mels = 224
    n_fft = 1024
    hop_length = n_fft // 2
    sample_rate = 32000
    frame_duration = 10.
    pad_remainder = True
    min_examples = 10
    max_examples = None
    valid_pct = 0.2
    random_seed = None

    @classmethod
    def from_dict(cls, **kwargs):
        instance = cls()
        instance.__dict__.update(**kwargs)
        return instance
