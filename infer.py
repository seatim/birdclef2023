
import os
import sys

from os.path import exists, join

import click
import numpy as np

from fastai.vision.all import load_learner, parent_label, Resize
from PIL import Image

from adak.config import TrainConfig
from adak.evaluate import avg_precision_over_subset, do_filter_top_k
from adak.transform import images_from_audio, image_width


def validate_paths(paths, model_classes):
    if not all(os.sep in path for path in paths):
        raise ValueError('paths must have parent directory as label')

    present_classes = set(parent_label(path) for path in paths)
    unknown_classes = present_classes - set(model_classes)
    known_classes = present_classes - unknown_classes

    if unknown_classes:
        print('W: some classes present in input are not known to model. These '
              'will be ignored.')

    return known_classes, unknown_classes


def get_images_from_audio(path, quick, quicker, config, resize):
    if not exists(path):
        path = join(config.audio_dir, path)

    if quicker:
        images = images_from_audio(path, config, 1)
    else:
        images = images_from_audio(path, config)
        if quick:
            images = images[:1]

    # NB: These operations are pretty fast.  One image flip takes about two
    # NB: usec, an image multiply takes about 83 usec, and a resize takes
    # NB: about 42 usec on a test machine.
    images = [np.flip(img, axis=0) for img in images]
    images = [np.uint8(255 * img) for img in images]
    images = [resize(img) for img in images]

    return images


def load_image(path):
    f = Image.open(path)
    img = np.array(f.getdata(), dtype=np.uint8).reshape(*reversed(f.size))
    f.close()
    return img


@click.command()
@click.argument('model_path')
@click.option('-a', '--audio-dir', default=TrainConfig.audio_dir,
              show_default=True)
@click.option('-q', '--quick', is_flag=True,
              help='infer only first image of each audio file')
@click.option('-Q', '--quicker', is_flag=True,
              help='infer only first image of each audio file and process '
                   'minimum amount of audio.  NB: this option results in '
                   'inference over a different set of test images as compared '
                   'to those inferred by the --quick option!!!  These images '
                   'are contrast- and brightness-altered variants of those '
                   'inferred by the --quick option.')
@click.option('-k', '--filter-top-k', type=int,
              help='drop n-k lowest probability predictions and renormalize '
                   'the rest')
@click.option('-K', '--top-k-filter-sweep', is_flag=True)
@click.option('-v', '--verbose', is_flag=True)
def main(model_path, audio_dir, quick, quicker, filter_top_k,
         top_k_filter_sweep, verbose):

    learn = load_learner(model_path)
    classes = np.array(learn.dls.vocab)
    resize = Resize(TrainConfig.n_mels)
    config = TrainConfig.from_dict(audio_dir=audio_dir)

    if filter_top_k is not None:
        if top_k_filter_sweep:
            sys.exit('E: choose either --filter-top-k or --top-k-filter-sweep')

        if not (0 < filter_top_k < len(classes)):
            sys.exit('E: filter_top_k must be > 0 and < n_classes')

    # NB: set frame_hop_length = frame_width.  Overlapping frames are good for
    # NB: training but a waste of time in this context.
    frame_width = image_width(
        config.frame_duration, config.sample_rate, config.hop_length)
    config.frame_hop_length = frame_width
    expected_img_size = (config.n_mels, frame_width)

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

        if path.endswith('.ogg'):
            images = get_images_from_audio(path, quick, quicker, config, resize)
        else:
            images = [load_image(path)]

        if not all(img.shape == expected_img_size for img in images):
            sys.exit(f'E: image size != {expected_img_size}: {path}, '
                     f'{list(img.shape for img in images)}')

        preds = np.stack([learn.predict(img)[2].numpy() for img in images])
        assert preds.shape[1] == len(classes), preds[0]

        if filter_top_k:
            preds = do_filter_top_k(preds, filter_top_k)

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

    if top_k_filter_sweep:
        for k in (3, 5, 13, 36, 98, 264):
            y_pred_k = [do_filter_top_k(pred, k) for pred in y_pred]
            ap_score = avg_precision_over_subset(
                np.vstack(y_pred_k), np.hstack(y_true), classes, known_classes)
            print(f'average precision score, k={k}: {ap_score:.3f}')
    else:
        ap_score = avg_precision_over_subset(
            np.vstack(y_pred), np.hstack(y_true), classes, known_classes)
        print(f'average precision score: {ap_score:.3f}')


if __name__ == '__main__':
    main()
