
import math
import os
import re
import sys
import time

from os.path import basename, exists, join

import click
import numpy as np
import pandas as pd

from fastai.vision.all import (load_learner, parent_label, Resize,
                               get_image_files)
from PIL import Image
from tabulate import tabulate

from adak.augment import HTrans, htrans_mat  # needed by some learners
from adak.config import InferenceConfig
from adak.evaluate import avg_precision_over_subset, calculate_n_top_n
from adak.transform import images_from_audio
from adak.transform import add_histeq

DEFAULT_PREDS_DIR = 'data/preds'


def get_model_version(path):
    match = re.match('birdclef-model-(\d+.\d+).pkl$', basename(path))
    if match:
        return match.groups()[0]


def get_val_dir_version(path):
    path = basename(path.rstrip('/'))
    if path.startswith('val_images.'):
        return path[11:]


class BaseCM:
    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        pass


class EnsembleLearner:
    ACCEPTANCE_THRESHOLDS = (0.5, 0.25)

    def __init__(self, model_paths, efficient=False):
        assert model_paths, 'need at least one model'
        self.learners = [load_learner(path) for path in model_paths]

        # find minimal vocabulary
        self.dls = self.learners[0].dls
        for learner in self.learners[1:]:
            if set(learner.dls.vocab) - set(self.dls.vocab) == set():
                self.dls = learner.dls

        self.indices = [None] * len(model_paths)
        self.diff_info = []
        self.efficient = efficient
        self.prediction_count = 0

        for k, learner in enumerate(self.learners):
            if self.dls.vocab == learner.dls.vocab:
                pass
            elif set(self.dls.vocab) - set(learner.dls.vocab) == set():
                self.indices[k] = [list(learner.dls.vocab).index(name)
                                   for name in self.dls.vocab]
            else:
                sys.exit('E: models have incompatible vocabularies')

    @classmethod
    def get_threshold(cls, k):
        if k >= len(cls.ACCEPTANCE_THRESHOLDS):
            return cls.ACCEPTANCE_THRESHOLDS[-1]
        else:
            return cls.ACCEPTANCE_THRESHOLDS[k]

    def no_bar(self):
        return BaseCM()

    def _predict(self, img, k):
        self.prediction_count += 1

        learn = self.learners[k]
        indices = self.indices[k]

        with learn.no_bar():
            if indices:
                return learn.predict(img)[2][indices]
            else:
                return learn.predict(img)[2]

    def _efficient_preds(self, img):
        preds = [self._predict(img, 0)]
        k = 0

        while max(preds[0]) < self.get_threshold(k):
            k += 1
            if k == len(self.learners):
                break
            preds.append(self._predict(img, k))

        return preds

    def predict(self, img):
        if self.efficient:
            preds = self._efficient_preds(img)
        else:
            preds = [self._predict(img, k) for k in range(len(self.learners))]

        for pred in preds[1:]:
            L0_diff = float(sum(abs(pred - preds[0])))
            self.diff_info.append((float(max(pred)), L0_diff))

        return None, None, sum(preds) / len(preds)


def validate_acceptance_thresholds(thresholds):
    try:
        thresholds = [float(x) for x in thresholds.split(',')]
    except ValueError:
        sys.exit(
            'E: acceptance thresholds must be comma-separated list of floats')

    if not all(0 < x < 1 for x in thresholds):
        sys.exit('E: acceptance thresholds must be between 0 and 1')

    return thresholds


def validate_paths(paths, model_classes):
    if not all(os.sep in path for path in paths):
        raise ValueError('paths must have parent directory as label')

    present_classes = set(parent_label(path) for path in paths)
    unknown_classes = present_classes - set(model_classes)
    known_classes = present_classes - unknown_classes

    if unknown_classes:
        print(f'W: {len(unknown_classes)} classes present in input are not '
              f'known to model. These will be ignored.')
        print('I: first five unknown classes:', list(unknown_classes)[:5])

    return known_classes


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
@click.argument('model_path', nargs=-1)
@click.option('-a', '--audio-dir', default=InferenceConfig.audio_dir,
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
@click.option('-s', '--save-preds', help='path to file to save preds to')
@click.option('-S', '--val-dir',
              help='validation directory; if name starts with "val_images." '
                   'preds will be saved to file named accordingly in preds '
                   'directory')
@click.option('-p', '--preds-dir', default=DEFAULT_PREDS_DIR,
              show_default=True)
@click.option('-e', '--efficient', is_flag=True)
@click.option('-t', '--acceptance-thresholds',
              help='comma-separated list of floats')
@click.option('-H', '--add-histeq', 'do_add_histeq', is_flag=True)
@click.option('-v', '--verbose', is_flag=True)
def main(model_path, audio_dir, quick, quicker, save_preds, val_dir, preds_dir,
         efficient, acceptance_thresholds, do_add_histeq, verbose):

    if acceptance_thresholds:
        thresholds = validate_acceptance_thresholds(acceptance_thresholds)
        EnsembleLearner.ACCEPTANCE_THRESHOLDS = thresholds

        if not efficient:
            efficient = True
            print('I: setting efficient = True due to use of '
                  '--acceptance-thresholds')
    else:
        acceptance_thresholds = ','.join(
            [str(x) for x in EnsembleLearner.ACCEPTANCE_THRESHOLDS])

    if val_dir:
        paths = [str(p) for p in get_image_files(val_dir)]

        model_version = '+'.join(get_model_version(p) for p in model_path)
        if efficient:
            model_version += f"-eff-{acceptance_thresholds}"

        val_dir_version = get_val_dir_version(val_dir)
        if do_add_histeq:
            val_dir_version += '-heq'

        if val_dir_version and model_version:
            fname = f'model-{model_version}.pkl-val.{val_dir_version}.csv'
            save_preds = join(preds_dir, fname)
    else:
        paths = None
        val_dir_version = ''

    if save_preds and exists(save_preds):
        sys.exit(f'E: file exists: {save_preds}')

    if len(model_path) > 1:
        learn = EnsembleLearner(model_path, efficient)
    else:
        learn = load_learner(model_path[0])
    classes = np.array(learn.dls.vocab)

    nse_val_dir = re.search('nse_\d+.\d+_0.\d+', val_dir_version)
    if nse_val_dir and 'NSE' not in classes:
        classes = np.array(list(classes) + ['NSE'])
        add_nse_column = True
    else:
        add_nse_column = False

    resize = Resize(InferenceConfig.n_mels)
    config = InferenceConfig.from_dict(audio_dir=audio_dir)
    expected_img_size = (config.n_mels, config.frame_width)

    n_inferences = n_top1 = n_top5 = 0

    # to calculate AP score we need to accumulate predictions and true values.
    y_pred = []
    y_true = []

    paths = paths or [line.strip() for line in sys.stdin]
    known_classes = validate_paths(paths, classes)
    last_time = time.time()

    for j, path in enumerate(paths):

        now = time.time()
        if now - last_time > 0.1:
            short_name = os.sep.join(path.split(os.sep)[-2:])
            print(f'Inferring {short_name} [{j}/{len(paths)}]...', end='\r',
                  flush=True)
            last_time = now

        y = parent_label(path)
        assert y in known_classes, y
        y_index = list(classes).index(y)

        if path.endswith('.ogg'):
            images = get_images_from_audio(path, quick, quicker, config, resize)
        else:
            images = [load_image(path)]

        if do_add_histeq:
            images = [add_histeq(img) for img in images]

        if not all(img.shape == expected_img_size for img in images):
            sys.exit(f'E: image size != {expected_img_size}: {path}, '
                     f'{list(img.shape for img in images)}')

        with learn.no_bar():
            preds = np.stack([learn.predict(img)[2].numpy() for img in images])

        if add_nse_column:
            preds = np.hstack([preds, np.zeros((1, len(images)))])

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
        n_top1 += sum(label == y for label in top1)
        n_top5 += sum(y in labels for labels in top5)

        y_pred.append(preds)
        y_true.append([y_index] * len(images))

    if not n_inferences:
        sys.exit('No inferences were made.')

    print('Results:')
    print(f'{n_inferences} inferences')
    print(f'{n_top1} top 1 correct {100 * n_top1 / n_inferences : .1f}%')
    print(f'{n_top5} top 5 correct {100 * n_top5 / n_inferences : .1f}%')

    y_pred = np.vstack(y_pred)
    y_true = np.hstack(y_true)

    assert n_top1 == calculate_n_top_n(y_pred, y_true, classes, 1)
    assert n_top5 == calculate_n_top_n(y_pred, y_true, classes, 5)

    def ap_score(y_pred):
        return avg_precision_over_subset(
            y_pred, y_true, classes, known_classes)

    best_ap_score = ap_score(y_pred)
    print(f'average precision score: {best_ap_score:.3f}')

    if save_preds:
        assert len(paths) == y_pred.shape[0], (len(paths), y_pred.shape)
        df = pd.DataFrame(dict(path=paths, **dict(zip(classes, y_pred.T))))
        df.to_csv(save_preds)

    if hasattr(learn, 'diff_info'):
        df = pd.DataFrame(learn.diff_info, columns=('max_pred', 'L0_diff'))
        print('Ensemble diff info:')
        print()
        print(df.head())
        print()
        print(df.describe().T)
        print()
        corr_matrix = np.corrcoef(list(zip(*df.values)))
        assert corr_matrix.shape == (2, 2), corr_matrix.shape
        a, b, c, d = corr_matrix.flatten()
        assert math.isclose(a, 1), a
        assert math.isclose(d, 1), d
        assert math.isclose(b, c), (b, c)
        print('correlation:', b)

        if efficient:
            print()
            print('prediction count:', learn.prediction_count)


if __name__ == '__main__':
    main()
