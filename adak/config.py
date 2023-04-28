

class BaseConfig:
    audio_dir = 'data/birdclef-2023/train_audio'
    images_dir = 'data/birdclef-2023/train_images'
    n_mels = 224
    n_fft = 1024
    hop_length = n_fft // 2
    sample_rate = 32000
    frame_duration = 5.
    frame_hop_factor = 2
    pad_remainder = True
    valid_pct = 0.2

    def image_width(self, audio_play_time):
        return 1 + int(audio_play_time * self.sample_rate) // self.hop_length

    @property
    def frame_width(self):
        return self.image_width(self.frame_duration)

    @property
    def frame_hop_length(self):
        return self.frame_width // self.frame_hop_factor

    @classmethod
    def from_dict(cls, **kwargs):
        instance = cls()
        instance.__dict__.update(**kwargs)
        return instance


class MakeImagesConfig(BaseConfig):
    min_examples_per_class = 10
    max_examples_per_class = 10000
    max_paths_per_class = None
    max_images_per_file = None
    sample_retries = 5


class TrainConfig(BaseConfig):
    bc21_images_dir = 'data/birdclef-2021/train_images'
    bc22_images_dir = 'data/birdclef-2022/train_images'
    combined_images_dir = 'data/train_images.combined'
    random_seed = None
    arch = 'efficientnet_b0'
    n_epochs = 5
    learn_rate = 0.01


class InferenceConfig(BaseConfig):
    frame_hop_factor = 1
