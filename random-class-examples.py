
import random

from os.path import join

import click
import pandas as pd

DEFAULT_DATA_DIR = 'data'


@click.command()
@click.option('-d', '--data-dir', type=click.Path(), default=DEFAULT_DATA_DIR,
              show_default=True)
@click.option('-v', '--verbose', is_flag=True)
def main(data_dir, verbose):
    tmd = pd.read_csv(join(data_dir, 'train_metadata.csv'))
    if verbose:
        class_counts = tmd.primary_label.value_counts()
        print('Number of classes:', len(class_counts))
        print('Class count min/mean/max/std:', class_counts.min(), '/',
              '%.1f' % class_counts.mean(), '/', class_counts.max(), '/',
              '%.1f' % class_counts.std())

    for k, (name, df) in enumerate(tmd.groupby('primary_label')):
        instance = random.choice(df.index)
        print(f'{k}. ({name}) {df.loc[instance].filename}')


if __name__ == '__main__':
    main()