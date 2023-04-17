
import sys

from os.path import exists, join

import click
import numpy as np

from fastai.vision.all import load_learner, parent_label, Resize
from torch import tensor

from adak.config import TrainConfig
from adak.glue import avg_precision
from adak.transform import images_from_audio, image_width


def validate_paths(paths, model_classes):
    if not all(len(path.split('/')) == 2 for path in paths):
        raise ValueError('audio paths must be relative to audio directory')

    present_classes = set(parent_label(path) for path in paths)
    unknown_classes = present_classes - set(model_classes)
    known_classes = present_classes - unknown_classes

    if unknown_classes:
        print('W: some classes present in input are not known to model. These '
              'will be ignored.')

    return known_classes, unknown_classes


def avg_precision_over_subset(y_pred, y_true, classes, subset):
    unknown_classes = set(subset) - set(classes)
    assert unknown_classes == set(), unknown_classes

    classes = list(classes)
    subset = list(subset)
    present_class_indices = [classes.index(label) for label in subset]

    assert len(y_pred.shape) == 2, y_pred.shape
    assert len(y_true.shape) == 1, y_true.shape
    n_samples = y_pred.shape[0]
    assert n_samples == len(y_true), (n_samples, len(y_true))
    assert y_pred.shape[1] == len(classes), (y_pred.shape, len(classes))
    assert all(idx in range(len(classes)) for idx in y_true)

    y_pred = np.array([pred[present_class_indices] for pred in y_pred])
    y_true = np.array([present_class_indices.index(idx) for idx in y_true])

    assert len(y_pred.shape) == 2, y_pred.shape
    assert len(y_true.shape) == 1, y_true.shape
    assert n_samples == len(y_true), (n_samples, len(y_true))
    assert n_samples == y_pred.shape[0], (n_samples, y_pred.shape)
    assert y_pred.shape[1] == len(subset), (y_pred.shape, len(subset))
    assert all(idx in range(len(subset)) for idx in y_true)

    return avg_precision(y_pred, tensor(y_true), len(subset))


@click.command()
@click.argument('model_path')
@click.option('-a', '--audio-dir', default=TrainConfig.audio_dir,
              show_default=True)
@click.option('-q', '--quick', is_flag=True,
              help='infer only first image of each audio file')
@click.option('-v', '--verbose', is_flag=True)
def main(model_path, audio_dir, quick, verbose):
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

    paths = [line.strip() for line in sys.stdin]
    known_classes, unknown_classes = validate_paths(paths, classes)

    for path in paths:
        correct_label = parent_label(path)
        if correct_label in unknown_classes:
            print(f'I: unknown class {correct_label}, skipping')
        correct_label_index = list(learn.dls.vocab).index(correct_label)

        if not exists(path):
            path = join(audio_dir, path)
        images = images_from_audio(path, config)
        if quick:
            images = images[:1]

        # NB: These operations are pretty fast.  One image flip takes about two
        # NB: usec, an image multiply takes about 83 usec, and a resize takes
        # NB: about 42 usec on a test machine.
        images = [np.flip(img, axis=0) for img in images]
        images = [np.uint8(255 * img) for img in images]
        images = [resize(img) for img in images]

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

    ap_score = avg_precision_over_subset(
        np.vstack(y_pred), np.hstack(y_true), classes, known_classes)
    print(f'average precision score: {ap_score:.3f}')


if __name__ == '__main__':
    main()
