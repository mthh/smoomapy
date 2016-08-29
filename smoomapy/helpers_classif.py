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
        return None

    def cut_head_tail_break(self, values):
        mean = values.mean()
        self.bins.append(mean)
        if len(values) > 1:
            return self.cut_head_tail_break(values[self.operator(values, mean)])
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
            diffs = diffs[-k1:]

        self.bins = np.array(
             [((sorted_copy[_id] + sorted_copy[_id + 1]) / 2.0)[0] for diff in diffs
              for _id in np.nonzero(d == diff)] + [sorted_copy[-1]]
            )
        self.bins.sort()


class SumsArray:
    def __init__(self, values):
        s = 0
        sumed_array = [0]
        for val in values:
            s += val
            sumed_array.append(s)
        self.sumed_array = sumed_array

    def Psum(self, i, j):
        return self.sumed_array[j] - self.sumed_array[i]

def dp_optimized_optimal_breaks(values, k=None, balance=0.5):
    """
    Naive implementation of pseudo-code from
    https://www.cs.umd.edu/sites/default/files/scholarly_papers/Abboud.pdf#page=9
    Compute breaks, allowing to balance between equal-area (balance=0)
    and equal-lenght/quantiles (balance=1)
    """
    if isinstance(values, (tuple, list)):
        values = np.array(values)
    elif not isinstance(values, np.ndarray):
        raise TypeError("Invalid type of data - Must be tuple, list or numpy.ndarray")
    n = len(values)
    SumsValues = SumsArray(values)
    Psum = SumsValues.Psum
    total = Psum(0, n)
    w = int(balance)
    inv_w = 1 - w
    if not k:
        k = get_opt_nb_class(n)
    avg = values.sum() / k
    avg_len = n / k
    best_error = np.empty((n+1, k))
    best_breaks = [[None]*(k) for _ in range(n+1)]
    for m in range(n):
        best_error[m][0] = inv_w * (((Psum(0, m) - avg) / total)**2) \
            + w * ((m - avg_len) / n) ** 2
        best_breaks[m] = list(map(lambda x: [], best_breaks[m]))
    for b in range(1, k - 1):
        for m in range(n + 1):
            min_error = best_break = None
            for _break in range(0, m):
                break_error = best_error[_break][b-1] \
                    + (inv_w * ((Psum(_break, m) - avg) / total) ** 2) \
                    + (w * (((m - _break) - avg_len) / n)**2)
                if not min_error or break_error < min_error:
                    min_error = break_error
                    best_break = _break
                    best_error[m][b] = min_error
                    best_breaks[m][b] = best_breaks[best_break][b-1] + [best_break]
    return sorted(values[np.unique(list(_chain(*best_breaks[n][1:k-1])))])
