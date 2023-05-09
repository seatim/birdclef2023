
from itertools import chain, combinations

import numpy as np
import pandas as pd


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
