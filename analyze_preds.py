
import sys

from os.path import dirname, join

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabulate import tabulate


def get_bc23_classes(path):
    tmd = pd.read_csv(join(dirname(path), '..', 'train_metadata.csv'))
    return set(tmd.primary_label)


def show_dist(series, desc, show_hist):
    print()
    print()
    print(f'Statistics of {desc}')
    print()
    stats = series.describe()
    print(tabulate([[int(stats[0])] + list(stats[1:])],
                   headers=list(stats.index), floatfmt='.3f'))

    if show_hist:
        ax = series.hist()
        ax.set_title(f'Histogram of {desc}')
        plt.show()


@click.command()
@click.argument('path')
@click.option('-s', '--show-hist', is_flag=True)
@click.option('-p', '--threshold', type=float)
def main(path, show_hist, threshold):

    if (threshold is not None) and not (0 < threshold < 1):
        sys.exit('E: threshold must be between 0 and 1.')

    df = pd.read_csv(path, index_col=0)
    classes = list(get_bc23_classes(path))
    assert set(classes) - set(df.columns) == set(), 'missing classes'

    df = df.set_index('path')
    assert sum(df.index.duplicated()) == 0, 'path column is not unique'

    show_dist(df.sum(axis=1), 'sum of predictions over all classes', show_hist)
    show_dist(df[classes].sum(axis=1), 'sum of predictions over bc23 classes',
              show_hist)

    show_dist(df.max(axis=1), 'max of predictions over all classes', show_hist)
    show_dist(df[classes].max(axis=1), 'max of predictions over bc23 classes',
              show_hist)

    if threshold:
        all_classes = np.array(df.columns)
        df['sum_bc23'] = df[classes].sum(axis=1)
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


if __name__ == '__main__':
    main()
