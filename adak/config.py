

class TrainConfig:
    audio_dir = 'data/birdclef-2023/train_audio'
    images_dir = 'data/birdclef-2023/train_images'
    bc21_images_dir = 'data/birdclef-2021/train_images'
    n_mels = 224
    n_fft = 1024
    hop_length = n_fft // 2
    sample_rate = 32000
    frame_duration = 5.
    frame_hop_length = (1 + int(frame_duration*sample_rate) // hop_length) // 2
    pad_remainder = True
    min_examples_per_class = 10
    max_examples_per_class = 10000
    max_paths_per_class = None
    max_images_per_file = None
    valid_pct = 0.2
    random_seed = None
    use_sed = False

    @classmethod
    def from_dict(cls, **kwargs):
        instance = cls()
        instance.__dict__.update(**kwargs)
        return instance
