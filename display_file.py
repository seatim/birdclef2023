
import click
import librosa
import matplotlib.pyplot as plt


@click.command()
@click.argument('path', type=click.Path())
def main(path):
    audio, sr = librosa.load(path, sr=None)
    assert len(audio.shape) == 1, audio.shape
    print('Sample rate:', sr)

    play_time = len(audio) / sr
    print('Playback length:', play_time)
    librosa.display.waveshow(audio, sr=sr)
    plt.show()


if __name__ == '__main__':
    main()
