#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some helpers functions to try some (one dimension) data classification methods.
"""
import numpy as np
from operator import ge, le
from math import floor, log10


def head_tail_breaks(values, direction="head"):
    return HeadTailBreaks(values, direction).bins


def maximal_breaks(values, k=None, diffmin=0):
    k = k if k else get_opt_nb_class(len(values))
    return [min(values)] + MaximalBreaks(values, k, diffmin).bins.tolist()


def get_opt_nb_class(len_values):
    return floor(1 + 3.3 * log10(len_values))


def _chain(*lists):
    for li in lists:
        for elem in li:
            yield elem


class HeadTailBreaks:
    def __init__(self, values, direction="head"):
        self.values = values if isinstance(values, np.ndarray) \
            else np.array(values)

        if "head" in direction:
            self.bins = [self.values.min()]
            self.operator = ge
        elif "tail" in direction:
            self.bins = [self.values.max()]
            self.operator = le
        else:
            raise ValueError(
                "Invalide direction argument (should be \"head\" or \"tail\")")

        self.cut_head_tail_break(self.values)
        self.nb_class = len(self.bins) - 1

        if "tail" in direction:
            self.bins = list(reversed(self.bins))

        return None

    def cut_head_tail_break(self, values):
        mean = values.mean()
        self.bins.append(mean)
        if len(values) > 1:
            return self.cut_head_tail_break(
                values[self.operator(values, mean)])
        return None


class MaximalBreaks:
    def __init__(self, values, k, diffmin=0):
        self.values = np.array(values)
        self.k = k
        self.diffmin = diffmin
        self.compute()
        self.nb_class = len(self.bins) - 1

    def compute(self):
        sorted_copy = self.values[:]
        sorted_copy.sort()
        d = sorted_copy[1:] - sorted_copy[:-1]
        diffs = np.unique(d[np.nonzero(d > self.diffmin)])
        k1 = self.k - 1
        if len(diffs) > k1:
            diffs = diffs[-int(k1):]

        self.bins = np.array(
             [((sorted_copy[_id] + sorted_copy[_id + 1]) / 2.0)[0]
              for diff in diffs
              for _id in np.nonzero(d == diff)] + [sorted_copy[-1]]
            )
        self.bins.sort()
