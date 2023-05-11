
import os

from collections import defaultdict
from itertools import chain, combinations

from os.path import basename, exists, join

import numpy as np
import pandas as pd

from fastai.data.all import parent_label

from .hashfile import file_sha1


def set_intersections(sets, digits=6):
    return [round(len(a & b) / len(a | b), digits)
            for a, b in combinations(sets, 2)]


class SoundEventDetectionFilter:
    COLUMNS = {'path', 'max_bc23', 'sha1'}

    def __init__(self, *paths, threshold=None):
        self.paths = list(paths)
        self.dfs = [pd.read_csv(p) for p in self.paths]
        self.threshold = threshold

        for p, df in zip(paths, self.dfs):
            assert set(df.columns) == self.COLUMNS, (p, df.columns)

    def examine(self):
        sha1s = [set(df.sha1) for df in self.dfs]
        fn_unique_hashes = [round(len(s) / len(df.sha1), 6)
                            for s, df in zip(sha1s, self.dfs)]
        print(f'I: there are {len(set(chain(*sha1s)))} unique hashes.')
        print(f'I: fraction of unique hashes, by file: {fn_unique_hashes}')

        if len(self.dfs) > 1:
            print(f'I: fraction of intersection of hashes, by file pair: '
                  f'{set_intersections(sha1s)}')

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value
        if value is None:
            self.ps = []
            self.nse = []
            return

        if not (0 <= value < 1):
            raise ValueError('threshold must be between 0 and 1')

        self.ps = [round(np.quantile(df['max_bc23'], value), 6)
                   for df in self.dfs]
        self.nse = [df[df['max_bc23'] < p] for df, p in zip(self.dfs, self.ps)]
        print(f'I: NSE thresholds for quantile {value} (max_bc23): {self.ps}')

        if len(self.dfs) > 1:
            print(f'I: fraction of intersection of hashes, by nse set pair: '
                  f'{set_intersections([set(x) for x in self.nse_hashes])}')

    @property
    def nse_hashes(self):
        return [set(df['sha1']) for df in self.nse]

    @property
    def nse_count(self):
        return [len(df['sha1']) for df in self.nse]

    def __contains__(self, sha1):
        return all([sha1 in hashes for hashes in self.nse_hashes])

    def relabel_files(self, dir_):
        nse_dir = join(dir_, 'NSE')
        os.makedirs(nse_dir, exist_ok=True)

        is_train = 'train' in basename(dir_)
        if not is_train:
            assert 'val' in basename(dir_), basename(dir_)

        # for a training dataset we need at least three examples per class for
        # the split, so keep track of the NSE examples here and do not relabel
        # any example until we know we have more than three.
        nse_hashes = defaultdict(set)
        good_hashes = defaultdict(set)
        nse_paths = defaultdict(list)

        for root, dirs, files in os.walk(dir_):

            for name in files:
                if not name.endswith('.png'):
                    continue

                path = join(root, name)
                label = parent_label(path)
                sha1 = file_sha1(path)

                if sha1 in self:
                    nse_hashes[label].add(sha1)
                    nse_paths[sha1].append(path)
                else:
                    good_hashes[label].add(sha1)

        nse_count = sum(len(s) for s in nse_hashes.values())
        print(f'I: identified {nse_count} NSE example hashes')

        move_count = 0
        save_count = 0

        for label, nse in nse_hashes.items():
            if is_train:
                nse = list(nse)
                np.random.shuffle(nse)

                n_good = len(good_hashes[label])
                while n_good < 3 and nse:
                    sha1 = nse.pop()
                    save_count += len(nse_paths[sha1])
                    n_good += 1

            while nse:
                sha1 = nse.pop()
                for path in nse_paths[sha1]:
                    os.rename(path, join(nse_dir, basename(path)))
                    move_count += 1

        print(f'I: relabeled {move_count} NSE example files')
        print(f'I: passed on relabeling {save_count} NSE example files')
