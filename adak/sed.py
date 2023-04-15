
from functools import partial
from itertools import chain

import numpy as np

from fastai.vision.all import parent_label, Callback
from scipy.special import expit
from tabulate import tabulate


def bind_alt(bound_method):
    """Work around fastcore's surprising handling of bound methods.  Fastcore
    calls bound methods as free functions with different objects passed as
    self.

    This workaround works because unlike bound methods, partial functions do
    not have a __func__ attribute.
    """
    return partial(bound_method.__func__, bound_method.__self__)


class SoundEventDetectionFilter(Callback):
    NON_EVENT = 'NON_EVENT'  # class label representing absence of sound event
    NON_EVENT_THRESHOLD = 0.1  # max prob. over classes

    def __init__(self):
        self.non_event_samples = set()
        self.preds = None

    def before_epoch(self):
        assert self.NON_EVENT in self.learn.dls.vocab

        all_items = set(chain(*[loader.items
                                for loader in self.learn.dls.loaders]))
        all_idxs = set(chain(*self.learn.dls.splits))
        assert len(all_items) == len(all_idxs)

        self.non_event_samples = set()
        self.preds = [[], []]  # train, val

    def after_batch(self):
        preds = self.preds[self.learn.dl.split_idx]
        preds.append(self.learn.pred.cpu().clone().detach())

    @staticmethod
    def print_stats(preds):
        for split_idx, split_name in enumerate(('train', 'val')):
            table = []

            for thresh in np.linspace(0.1, 0.9, 9):
                # for each sample, find the max. prob over classes and compare
                # that value to the threshold
                max_signal = np.max(preds[split_idx], axis=1)
                table.append((thresh, sum(max_signal < thresh)))

            print(f'num. signals statistics for {split_name} split:')
            print(tabulate(table, headers=('threshold', 'n_non_event')))

    def after_epoch(self):
        preds = [expit(np.vstack(p)) for p in self.preds]
        self.print_stats(preds)

        for loader, p, short in zip(self.learn.dls.loaders,
                                    preds, (True, False)):
            if short:
                assert len(p) == loader.bs * (len(loader.items) // loader.bs)
            else:
                assert len(p) == len(loader.items)

            max_signal = np.max(p, axis=1)
            for value, item in zip(max_signal, loader.items):
                if value < self.NON_EVENT_THRESHOLD:
                    print(f'NON EVENT:', item)
                    self.non_event_samples.add(str(item))

    def get_y(self, path):
        if str(path) in self.non_event_samples:
            return self.NON_EVENT
        else:
            return parent_label(path)
