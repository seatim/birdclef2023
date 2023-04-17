
import sys
import warnings

from os.path import exists, join

import click
import numpy as np

from fastai.vision.all import load_learner, parent_label, Resize
from torch import tensor

from adak.config import TrainConfig
from adak.glue import avg_precision
from adak.transform import images_from_audio, image_width


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

    # NB: set frame_hop_length = frame_width.  Overlapping frames are good for
    # NB: training but a waste of time in this context.
    frame_width = image_width(
        config.frame_duration, config.sample_rate, config.hop_length)
    config.frame_hop_length = frame_width

    n_inferences = n_top1 = n_top5 = 0

    # to calculate AP score we need to accumulate predictions and true values.
    y_pred = []
    y_true = []

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
        correct_label_index = list(learn.dls.vocab).index(correct_label)

        preds = np.stack([learn.predict(img)[2].numpy() for img in images])
        assert preds.shape[1] == len(classes), preds[0]

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

        y_pred.append(preds)
        y_true.append([correct_label_index] * len(images))

    print('Results:')
    print(f'{n_inferences} inferences')
    print(f'{n_top1} top 1 correct {100 * n_top1 / n_inferences : .1f}%')
    print(f'{n_top5} top 5 correct {100 * n_top5 / n_inferences : .1f}%')

    warnings.filterwarnings(
        action='ignore', category=UserWarning,
        message='No positive class found in y_true, recall')

    ap_score = avg_precision(
        np.vstack(y_pred), tensor(np.hstack(y_true)), len(classes))
    print(f'average precision score: {ap_score:.3f}')


if __name__ == '__main__':
    main()
