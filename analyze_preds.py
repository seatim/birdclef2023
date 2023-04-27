
import os
import sys

from os.path import dirname, join

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastai.data.all import parent_label
from tabulate import tabulate

from adak.evaluate import (avg_precision_over_subset, calculate_n_top_n,
                           do_filter_top_k, fine_threshold, sum_filter,
                           max_filter, slice_by_class_subset)


def short_name(row):
    return os.sep.join(row.path.split(os.sep)[-2:])


def get_bc23_classes(path):
    tmd = pd.read_csv(join(dirname(path), '..', 'train_metadata.csv'))
    return set(tmd.primary_label)


def get_classes_and_y_vars(df):
    classes = list(df.columns)
    y_pred = df[classes].to_numpy()
    y_true = np.array([classes.index(name)
                       for name in (parent_label(path) for path in df.index)])

    missing = set(range(len(classes))) - set(y_true)
    missing_classes = [classes[k] for k in sorted(missing)]
    return classes, missing_classes, y_pred, y_true


def report_essentials(df, bc23_classes):
    n_inferences = len(df.index)
    classes, missing_classes, y_pred, y_true = get_classes_and_y_vars(df)

    if missing_classes:
        print(f'W: missing examples for {len(missing_classes)} classes.')
        print(f'W: first five are:', missing_classes[:5])

    n_top1 = calculate_n_top_n(y_pred, y_true, classes, 1)
    n_top5 = calculate_n_top_n(y_pred, y_true, classes, 5)
    ap_score = avg_precision_over_subset(
        y_pred, y_true, classes, set(classes) - set(missing_classes))

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

    if set(classes) != set(bc23_classes):
        y_pred_b, y_true_b = slice_by_class_subset(
            y_pred, y_true, classes, bc23_classes)

        n_top1_b = calculate_n_top_n(y_pred_b, y_true_b, bc23_classes, 1)
        n_top5_b = calculate_n_top_n(y_pred_b, y_true_b, bc23_classes, 5)

        ap_score_b = avg_precision_over_subset(
            y_pred, y_true, classes, set(bc23_classes) - set(missing_classes))

        print('Results for bc23 classes:')
        print(f'{n_top1_b} top 1 correct {100 * n_top1_b / n_inferences:.1f}%')
        print(f'{n_top5_b} top 5 correct {100 * n_top5_b / n_inferences:.1f}%')
        print(f'average precision score: {ap_score_b:.3f}')
        print()


def sweep_preds_AP_score(y_pred, ap_score, values, param_name, func, desc):
    y_preds = [func(y_pred, x) for x in values]
    ap_scores = [ap_score(y_pred_x) for y_pred_x in y_preds]

    print()
    print(f'AP scores for {desc}:')
    print()
    print(tabulate([ap_scores], headers=values))
    print()


def report_sweeps(df, bc23_classes):
    classes, missing_classes, y_pred, y_true = get_classes_and_y_vars(df)

    def ap_score(y_pred):
        return avg_precision_over_subset(
            y_pred, y_true, classes, set(classes) - set(missing_classes))

    ks = (3, 5, 13, 36, 98, 264)
    sweep_preds_AP_score(
        y_pred, ap_score, ks, 'k', do_filter_top_k, 'top-k filter')

    ps = list(reversed((1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.9)))
    sweep_preds_AP_score(
        y_pred, ap_score, ps, 'p', fine_threshold, 'fine threshold')

    sweep_preds_AP_score(
        y_pred, ap_score, ps, 'p', sum_filter, 'sum filter')

    sweep_preds_AP_score(
        y_pred, ap_score, ps, 'p', max_filter, 'max filter')

    if set(classes) != set(bc23_classes):
        y_pred_b, y_true_b = slice_by_class_subset(
            y_pred, y_true, classes, bc23_classes)

        def ap_score_b(y_pred):
            return avg_precision_over_subset(
                y_pred, y_true_b, bc23_classes,
                set(bc23_classes) - set(missing_classes))

        sweep_preds_AP_score(
            y_pred_b, ap_score_b, ks, 'k', do_filter_top_k,
            'top-k filter over bc23 classes')

        sweep_preds_AP_score(
            y_pred_b, ap_score_b, ps, 'p', fine_threshold,
            'fine threshold over bc23 classes')

        sweep_preds_AP_score(
            y_pred_b, ap_score_b, ps, 'p', sum_filter,
            'sum filter over bc23 classes')

        sweep_preds_AP_score(
            y_pred_b, ap_score_b, ps, 'p', max_filter,
            'max filter over bc23 classes')


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
    classes, missing_classes, y_pred, y_true = get_classes_and_y_vars(df)
    stats = [(name, np.mean(preds), max(preds))
             for name, preds in zip(classes, y_pred.T)]
    print()
    print('Class statistics:')
    print()
    print(tabulate(stats, headers=('name', 'mean', 'max')))
    print()

    show_dist(df.max(axis=0), 'max of predictions over examples', show_hist)


@click.command()
@click.argument('path')
@click.option('-s', '--show-hist', is_flag=True)
@click.option('-S', '--show-stats', is_flag=True)
@click.option('-r', '--report-sweeps', 'do_sweeps', is_flag=True)
@click.option('-R', '--report-class-stats', 'do_class_stats', is_flag=True)
@click.option('-p', '--threshold', type=float)
@click.option('-e', '--list-nse-candidates', is_flag=True)
def main(path, show_hist, show_stats, do_sweeps, do_class_stats, threshold,
         list_nse_candidates):

    if (threshold is not None) and not (0 < threshold < 1):
        sys.exit('E: threshold must be between 0 and 1.')

    df = pd.read_csv(path, index_col=0)
    bc23_classes = list(get_bc23_classes(path))
    assert set(bc23_classes) - set(df.columns) == set(), 'missing bc23 classes'

    df['short_name'] = df.apply(short_name, axis=1)
    df = df.drop('path', axis=1)
    df = df.set_index('short_name')
    assert sum(df.index.duplicated()) == 0, 'short_name column is not unique'

    report_essentials(df, bc23_classes)

    if do_sweeps:
        report_sweeps(df, bc23_classes)

    if show_stats:
        all_classes = np.array(df.columns)
        qualifier = 'all ' if set(all_classes) != set(bc23_classes) else ''

        show_dist(df.sum(axis=1),
                  f'sum of predictions over {qualifier}classes', show_hist)
        if set(all_classes) != set(bc23_classes):
            show_dist(df[bc23_classes].sum(axis=1),
                      'sum of predictions over bc23 classes', show_hist)

        show_dist(df.max(axis=1),
                  f'max of predictions over {qualifier}classes', show_hist)
        if set(all_classes) != set(bc23_classes):
            show_dist(df[bc23_classes].max(axis=1),
                      'max of predictions over bc23 classes', show_hist)

    if do_class_stats:
        report_class_stats(df, show_hist)

    if threshold:
        all_classes = np.array(df.columns)
        df['sum_bc23'] = df[bc23_classes].sum(axis=1)
        lp = df[df['sum_bc23'] < threshold]
        table = [[path, row['sum_bc23']] for path, row in lp.iterrows()]

        print()
        print('Examples with sum of bc23 class predictions below threshold:')
        print(tabulate(table, headers=['path', 'sum_bc23']))

        print()
        print('Top five predictions for the first of these examples:')
        for path, _ in table[:5]:
            print(path)
            row = df.loc[path]
            top5 = list(all_classes[row[all_classes].argsort()[-5:]])
            top5preds = list(row[top5])
            print(tabulate([top5 + [f'{p:.4f}' for p in top5preds]]))
            print()

    if list_nse_candidates:
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


if __name__ == '__main__':
    main()
