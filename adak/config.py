

class TrainConfig:
    audio_dir = 'data/train_audio'
    images_dir = 'data/train_images'
    model_dir = None
    n_mels = 224
    n_fft = 1024
    hop_length = n_fft // 2
    sample_rate = 32000
    frame_duration = 5.
    frame_hop_length = (1 + int(frame_duration*sample_rate) // hop_length) // 2
    pad_remainder = True
    min_examples = 10
    max_examples = None
    valid_pct = 0.2
    random_seed = None
    use_sed = False

    @classmethod
    def from_dict(cls, **kwargs):
        instance = cls()
        instance.__dict__.update(**kwargs)
        return instance
