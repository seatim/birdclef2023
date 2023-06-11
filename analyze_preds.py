
"""Analyze the predictions made by a classifier.
"""

import os
import re
import sys
import time
import warnings

from operator import itemgetter
from os.path import dirname, join

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastai.data.all import parent_label
from tabulate import tabulate

from adak.config import BaseConfig
from adak.evaluate import (avg_precision_over_subset, count_top_n,
                           slice_by_class_subset)
from adak.filter import top_k_filter, fine_filter, max_filter
from adak.hashfile import file_sha1


def short_path(path):
    return os.sep.join(path.split(os.sep)[-2:])


def short_name(row):
    return short_path(row.path)


def get_bc23_classes(path):
    tmd = pd.read_csv(join(dirname(path), 'train_metadata.csv'))
    return set(tmd.primary_label)


def get_classes_and_y_vars(df):
    classes = list(df.columns)
    y_pred = df[classes].to_numpy()
    y_true = np.array([classes.index(name)
                       for name in (parent_label(path) for path in df.index)])

    missing = set(range(len(classes))) - set(y_true)
    missing_classes = [classes[k] for k in sorted(missing)]
    return classes, missing_classes, y_pred, y_true


def add_df_attrs(df):
    attrs = 'classes', 'missing_classes', 'y_pred', 'y_true'
    assert all(not hasattr(df, x) for x in attrs), set(dir(df)) & set(attrs)
    df.classes, df.missing_classes, df.y_pred, df.y_true = \
        get_classes_and_y_vars(df)


def report_essentials(df, bc23_classes, do_bc23):
    n_inferences = len(df.index)
    classes, missing_classes, y_pred, y_true = \
        df.classes, df.missing_classes, df.y_pred, df.y_true

    if missing_classes:
        print(f'W: missing examples for {len(missing_classes)} classes.')
        print(f'W: first five are:', missing_classes[:5])

    n_top1 = count_top_n(y_pred, y_true, classes, 1)
    n_top5 = count_top_n(y_pred, y_true, classes, 5)
    ap_score = avg_precision_over_subset(
        y_pred, y_true, classes, set(classes) - set(missing_classes))
    ap_score_b = None

    print()
    if set(classes) == set(bc23_classes):
        print('Results:')
        print(f'{n_inferences} inferences')
    else:
        print(f'{n_inferences} inferences')
        print()
        print('Results for all classes:')

    print(f'{n_top1} top 1 correct {100 * n_top1 / n_inferences : .1f}%')
    print(f'{n_top5} top 5 correct {100 * n_top5 / n_inferences : .1f}%')
    print(f'average precision score: {ap_score:.3f}')
    print()

    if do_bc23:
        y_pred_b, y_true_b = slice_by_class_subset(
            y_pred, y_true, classes, bc23_classes)

        n_top1_b = count_top_n(y_pred_b, y_true_b, bc23_classes, 1)
        n_top5_b = count_top_n(y_pred_b, y_true_b, bc23_classes, 5)

        ap_score_b = avg_precision_over_subset(
            y_pred, y_true, classes, set(bc23_classes) - set(missing_classes))

        print('Results for bc23 classes:')
        print(f'{n_top1_b} top 1 correct {100 * n_top1_b / n_inferences:.1f}%')
        print(f'{n_top5_b} top 5 correct {100 * n_top5_b / n_inferences:.1f}%')
        print(f'average precision score: {ap_score_b:.3f}')
        print()

    return ap_score, ap_score_b


def sweep_preds_AP_score(y_pred, ap_score, values, param_name, func, desc,
                         best_ap_score, verbose):

    y_preds = [func(y_pred, x) for x in values]
    ap_scores = [ap_score(y_pred_x) for y_pred_x in y_preds]
    beats = ['^^^^^' if score > best_ap_score else '' for score in ap_scores]

    if any(beats):
        table = [['%.3f' % score for score in ap_scores], beats]
    else:
        table = [ap_scores]

    if verbose:
        print()
        print(f'AP scores for {desc}:')
        print()
        print(tabulate(table, headers=values, floatfmt='.3f'))
        print()

    elif any(beats):
        i = np.array(ap_scores).argmax()
        pc_better = 100 * (ap_scores[i] - best_ap_score) / best_ap_score
        print(f'AP score for {desc}, {param_name}={values[i]}: '
              f'{ap_scores[i]:.3f} ({pc_better:.1f} % better)')
    else:
        print(f'no better AP scores were found by {desc}')


def report_sweeps(df, bc23_classes, do_bc23, best_ap_score, best_ap_score_b,
                  verbose):

    classes, missing_classes, y_pred, y_true = \
        df.classes, df.missing_classes, df.y_pred, df.y_true

    def ap_score(y_pred):
        return avg_precision_over_subset(
            y_pred, y_true, classes, set(classes) - set(missing_classes))

    ks = (98, 137, 201, 233, 249, 257, 261, 263, 264)
    sweep_preds_AP_score(
        y_pred, ap_score, ks, 'k', top_k_filter, 'top-k filter',
        best_ap_score, verbose)

    ftps = (1e-7, 1e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3)
    sweep_preds_AP_score(
        y_pred, ap_score, ftps, 'p', fine_filter, 'fine filter',
        best_ap_score, verbose)

    mfps = [0.01 * x for x in range(1, 11)]
    sweep_preds_AP_score(
        y_pred, ap_score, mfps, 'p', max_filter, 'max filter', best_ap_score,
        verbose)

    if do_bc23:
        y_pred_b, y_true_b = slice_by_class_subset(
            y_pred, y_true, classes, bc23_classes)

        def ap_score_b(y_pred):
            return avg_precision_over_subset(
                y_pred, y_true_b, bc23_classes,
                set(bc23_classes) - set(missing_classes))

        sweep_preds_AP_score(
            y_pred_b, ap_score_b, ks, 'k', top_k_filter,
            'top-k filter over bc23 classes', best_ap_score_b, verbose)

        sweep_preds_AP_score(
            y_pred_b, ap_score_b, ftps, 'p', fine_filter,
            'fine filter over bc23 classes', best_ap_score_b, verbose)

        sweep_preds_AP_score(
            y_pred_b, ap_score_b, mfps, 'p', max_filter,
            'max filter over bc23 classes', best_ap_score_b, verbose)

    print()


def show_dist(series, desc, show_hist):
    print()
    print(f'Statistics of {desc}')
    print()
    stats = series.describe()
    print(tabulate([[int(stats[0])] + list(stats[1:])],
                   headers=list(stats.index), floatfmt='.3f'))
    print()

    if show_hist:
        ax = series.hist()
        ax.set_title(f'Histogram of {desc}')
        plt.show()


def report_class_stats(df, show_hist):
    classes, missing_classes, y_pred, y_true = \
        df.classes, df.missing_classes, df.y_pred, df.y_true

    stats = [(name, np.mean(preds), max(preds))
             for name, preds in zip(classes, y_pred.T)]

    stats_df = pd.DataFrame(dict(zip(('name', 'mean', 'max'), zip(*stats))))
    stats_df['n_true'] = pd.Series(
        [list(y_true).count(k) for k in range(len(classes))])
    stats_df = stats_df[~stats_df.name.isin(missing_classes)]

    print()
    print('Highest confidence classes:')
    print()
    print(stats_df.sort_values(['max', 'mean'], ascending=False).head(10))
    print()
    print('Lowest confidence classes:')
    print()
    print(stats_df.sort_values(['max', 'mean'], ascending=False).tail(10))
    print()

    show_dist(df.max(axis=0), 'max of predictions over examples', show_hist)


def hash_files(df):
    print()
    print('I: hashing files, this may take a while...')

    _a = os.environ.get('REWRITE_PATH_FROM')
    _b = os.environ.get('REWRITE_PATH_TO')

    if _a is not None and _b is not None:
        def rewrite_path(path):
            return re.sub(_a, _b, path)
    else:
        def rewrite_path(path):
            return path

    sha1s = []
    last_time = time.time()
    n_files = len(df.index)

    for k, path in enumerate(df['path']):
        now = time.time()
        if now - last_time > 0.1:
            last_time = now
            msg = f'I: hashing {short_path(path)} [{k}/{n_files}] ...   '
            print(msg, end='\r', flush=True)

        sha1s.append(file_sha1(rewrite_path(path)))

    df['sha1'] = sha1s
    print()


@click.command(help=__doc__)
@click.argument('path')
@click.option('-s', '--show-hist', is_flag=True)
@click.option('-S', '--show-stats', is_flag=True)
@click.option('-r', '--report-sweeps', 'do_sweeps', is_flag=True)
@click.option('-R', '--report-class-stats', 'do_class_stats', is_flag=True)
@click.option('-p', '--threshold', type=float)
@click.option('-e', '--list-nse-candidates', is_flag=True)
@click.option('-k', '--skip-bc23-classes', is_flag=True)
@click.option('-m', '--make-nse-file', is_flag=True)
@click.option('-n', '--nse-file-path')
@click.option('-v', '--verbose', is_flag=True)
def main(path, show_hist, show_stats, do_sweeps, do_class_stats, threshold,
         list_nse_candidates, skip_bc23_classes, make_nse_file, nse_file_path,
         verbose):

    if (threshold is not None) and not (0 < threshold < 1):
        sys.exit('E: threshold must be between 0 and 1.')

    preds_path = path
    df = pd.read_csv(path, index_col=0)
    bc23_classes = list(get_bc23_classes(BaseConfig.audio_dir))
    if set(bc23_classes) - set(df.columns):
        if not skip_bc23_classes:
           sys.exit('E: missing bc23 classes.  To proceed with analysis '
                    'anyway use the --skip-bc23-classes option.')

    df['short_name'] = df.apply(short_name, axis=1)
    df = df.set_index('short_name')
    assert sum(df.index.duplicated()) == 0, 'short_name column is not unique'

    path_column = df['path']
    df = df.drop('path', axis=1)

    all_classes = np.array(df.columns)
    do_bc23 = (set(all_classes) != set(bc23_classes)) and not skip_bc23_classes

    warnings.filterwarnings(
        action='ignore', category=UserWarning,
        message='Pandas doesn\'t allow columns to be created via a new'
    )
    add_df_attrs(df)

    ap_score, ap_score_b = report_essentials(df, bc23_classes, do_bc23)

    if do_sweeps:
        report_sweeps(df, bc23_classes, do_bc23, ap_score, ap_score_b, verbose)

    if show_stats:
        qualifier = 'all ' if set(all_classes) != set(bc23_classes) else ''

        def show_stat_over_classes(stat, qualifier):
            desc = stat.__name__ + ' of predictions over {}classes'
            show_dist(stat(axis=1), desc.format(qualifier), show_hist)

        show_stat_over_classes(df.sum, qualifier)
        if do_bc23:
            show_stat_over_classes(df[bc23_classes].sum, 'bc23 ')

        show_stat_over_classes(df.max, qualifier)
        if do_bc23:
            show_stat_over_classes(df[bc23_classes].max, 'bc23 ')

    if do_class_stats:
        report_class_stats(df, show_hist)

    if threshold and not skip_bc23_classes:
        df['max_bc23'] = df[bc23_classes].max(axis=1)
        lp = df[df['max_bc23'] < threshold]
        table = [[path, row['max_bc23']] for path, row in lp.iterrows()]
        table = sorted(table, key=itemgetter(1), reverse=True)

        print()
        print('Examples with max of bc23 class predictions below threshold:')
        print()
        print('Top five:')
        print(tabulate(table[:5], headers=['path', 'max_bc23']))
        print()
        if len(table) > 5:
            print('Bottom five:')
            print(tabulate(table[-5:], headers=['path', 'max_bc23']))
            print()

        print()
        print('Top five predictions for the first of these examples:')
        for path, _ in table[:5]:
            print(path)
            row = df.loc[path]
            top5 = list(all_classes[row[all_classes].argsort()[-5:]])
            top5preds = list(row[top5])
            print(tabulate([top5 + [f'{p:.4f}' for p in top5preds]]))
            print()

    if list_nse_candidates and not skip_bc23_classes:
        df['sum_bc23'] = df[bc23_classes].sum(axis=1)
        df['max_bc23'] = df[bc23_classes].max(axis=1)

        print()
        print('Examples with lowest sum of bc23 predictions:')
        print(df['sum_bc23'].sort_values().head(5))
        print()
        print()
        print('Examples with lowest max of bc23 predictions:')
        print(df['max_bc23'].sort_values().head(5))
        print()

    if make_nse_file and not skip_bc23_classes:
        df['max_bc23'] = df[bc23_classes].max(axis=1)
        df_nse = df[['max_bc23']].copy()
        df_nse['path'] = path_column

        hash_files(df_nse)
        output_path = nse_file_path or preds_path + '.nsedata.csv'
        df_nse.to_csv(output_path, index=False)
        print(f'I: wrote the NSE data to {output_path}')


if __name__ == '__main__':
    main()
