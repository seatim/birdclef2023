
import os
import sys

from os.path import exists, join

import click
import numpy as np

from fastai.vision.all import load_learner, parent_label, Resize
from PIL import Image
from tabulate import tabulate

from adak.config import TrainConfig
from adak.evaluate import (avg_precision_over_subset, do_filter_top_k,
                           apply_threshold)
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
@click.option('-K', '--no-top-k-filter-sweep', is_flag=True)
@click.option('-P', '--no-threshold-sweep', is_flag=True)
@click.option('-v', '--verbose', is_flag=True)
def main(model_path, audio_dir, quick, quicker, no_top_k_filter_sweep,
         no_threshold_sweep, verbose):

    learn = load_learner(model_path)
    classes = np.array(learn.dls.vocab)
    resize = Resize(TrainConfig.n_mels)
    config = TrainConfig.from_dict(audio_dir=audio_dir)

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

        top5 = [classes[pred.argsort()[-5:]] for pred in preds]
        top1 = [_[-1] for _ in top5]
        if verbose:
            print(f'top five predictions for {path} ({len(preds)} frames):')
            top5preds = [pred[pred.argsort()[-5:]] for pred in preds]
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

    def ap_score(y_pred):
        return avg_precision_over_subset(
            np.vstack(y_pred), np.hstack(y_true), classes, known_classes)

    best_ap_score = ap_score(y_pred)
    print(f'average precision score: {best_ap_score:.3f}')

    if not no_top_k_filter_sweep:
        ks = (3, 5, 13, 36, 98, 264)

        y_pred_ks = [[do_filter_top_k(pred, k) for pred in y_pred] for k in ks]
        ap_scores = [ap_score(y_pred_k) for y_pred_k in y_pred_ks]

        if any(score > best_ap_score for k, score in zip(ks, ap_scores)):
            i = np.array(ap_scores).argmax()
            print(f'average precision score, k={ks[i]}: {ap_scores[i]:.3f}')
        else:
            print('no better AP scores were found by top-k filtering')

    if not no_threshold_sweep:
        ps = (1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.9)

        y_pred_ps = [[apply_threshold(pred, p) for pred in y_pred] for p in ps]
        ap_scores = [ap_score(y_pred_p) for y_pred_p in y_pred_ps]

        if any(score > best_ap_score for p, score in zip(ps, ap_scores)):
            i = np.array(ap_scores).argmax()
            print(f'average precision score, p={ps[i]}: {ap_scores[i]:.3f}')
        else:
            print('no better AP scores were found by thresholding')


if __name__ == '__main__':
    main()
