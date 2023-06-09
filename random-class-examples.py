
import random

from os.path import join

import click
import numpy as np
import pandas as pd


@click.command()
@click.option('-d', '--data-dir', type=click.Path(), default='data',
              show_default=True)
@click.option('-n', '--num-examples-per-class', 'num_examples', default=1,
              show_default=True)
@click.option('-v', '--verbose', is_flag=True)
def main(data_dir, num_examples, verbose):
    tmd = pd.read_csv(join(data_dir, 'train_metadata.csv'))

    if verbose:
        class_counts = tmd.primary_label.value_counts()
        print('Number of classes:', len(class_counts))
        print('Class count min/mean/max/std:', class_counts.min(), '/',
              '%.1f' % class_counts.mean(), '/', class_counts.max(), '/',
              '%.1f' % class_counts.std())

    for label, df in tmd.groupby('primary_label'):
        instances = np.random.choice(df.index, size=num_examples)
        for idx, row in df.loc[instances].iterrows():
            if label not in row.filename:
                print(join(label, row.filename))
            else:
                print(row.filename)


if __name__ == '__main__':
    main()
