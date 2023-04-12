
import sys

from os.path import exists, join

import click
import numpy as np

from fastai.vision.all import load_learner, parent_label, Resize

from adak.config import TrainConfig
from adak.transform import images_from_audio


@click.command()
@click.argument('model_path')
@click.option('-a', '--audio-dir', default=TrainConfig.audio_dir,
              show_default=True)
@click.option('-v', '--verbose', is_flag=True)
def main(model_path, audio_dir, verbose):
    learn = load_learner(model_path)
    classes = np.array(learn.dls.vocab)
    resize = Resize(TrainConfig.n_mels)
    config = TrainConfig.from_dict(audio_dir=audio_dir)

    n_inferences = n_top1 = n_top5 = 0

    for line in sys.stdin:
        path = line.strip()
        if not exists(path):
            path = join(audio_dir, path)
        images = images_from_audio(path, config)

        # FIXME add flip and scale options to load function
        images = [np.flip(img, axis=0) for img in images]
        images = [np.uint8(255 * img) for img in images]
        images = [resize(img) for img in images]

        correct_label = parent_label(path)

        preds = [np.array(learn.predict(img)[2]) for img in images]
        assert np.array(preds).shape[1] == len(classes), preds[0]

        top5 = [classes[pred.argsort()[-5:]] for pred in preds]
        top1 = [_[-1] for _ in top5]
        if verbose:
            print(f'top five predictions for {path} ({len(preds)} frames):')
            top5preds = [pred[pred.argsort()[-5:]] for pred in preds]
            from tabulate import tabulate
            print(tabulate([[*a, *b] for a, b in zip(top5, top5preds)],
                           floatfmt=".2f"))
            print()

        n_inferences += len(images)
        n_top1 += sum(label == correct_label for label in top1)
        n_top5 += sum(correct_label in labels for labels in top5)

    print('Results:')
    print(f'{n_inferences} inferences')
    print(f'{n_top1} top 1 correct {100 * n_top1 / n_inferences : .1f}%')
    print(f'{n_top5} top 5 correct {100 * n_top5 / n_inferences : .1f}%')


if __name__ == '__main__':
    main()
