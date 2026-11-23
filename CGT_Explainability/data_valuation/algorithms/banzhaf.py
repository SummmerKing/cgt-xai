import numpy as np
import math
import logging
from itertools import chain, combinations
from .least_core import ValuationResult

logger = logging.getLogger(__name__)

def powerset(iterable):

    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))

def exact_banzhaf_index(utility, normalized=True, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)
    banzhaf_scores = np.zeros(n)
    players = range(n)

    for i in players:
        total_influence = 0.0
        for S in powerset([p for p in players if p != i]):
            S = list(S)
            marginal = utility(S + [i]) - utility(S)
            total_influence += marginal
        banzhaf_scores[i] = total_influence

    if normalized:
        total = np.sum(banzhaf_scores)
        if total > 0:
            banzhaf_scores /= total

    return ValuationResult(
        algorithm="exact_banzhaf_index",
        status="Converged",
        values=banzhaf_scores,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )

def montecarlo_banzhaf_index(utility, n_iterations=1000, normalized=True, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)
    banzhaf_scores = np.zeros(n)
    players = np.arange(n)

    for _ in range(n_iterations):
        coalition = np.random.choice([0, 1], size=n).astype(bool)
        coalition_indices = np.where(coalition)[0].tolist()

        for i in players:
            if i in coalition_indices:
                without_i = [p for p in coalition_indices if p != i]
                marginal = utility(coalition_indices) - utility(without_i)
            else:
                marginal = utility(coalition_indices + [i]) - utility(coalition_indices)
            banzhaf_scores[i] += marginal

    if normalized:
        total = np.sum(banzhaf_scores)
        if total > 0:
            banzhaf_scores /= total

    return ValuationResult(
        algorithm="montecarlo_banzhaf_index",
        status="Converged",
        values=banzhaf_scores,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )
