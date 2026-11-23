import numpy as np
import logging
from itertools import chain, combinations
from .least_core import ValuationResult

logger = logging.getLogger(__name__)

def powerset(iterable):
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))

def is_minimal_winning(coalition, utility, threshold):
    if utility(coalition) < threshold:
        return False
    for i in coalition:
        subcoalition = [j for j in coalition if j != i]
        if utility(subcoalition) >= threshold:
            return False
    return True

def exact_deegan_packel_index(utility, threshold=None, normalized=True, options=None, progress=False):
    if options is None:
        options = {}

    n = len(utility.data)
    players = range(n)
    dp_scores = np.zeros(n)

    if threshold is None:
        threshold = utility(list(players)) / 2.0


    for coalition in powerset(players):
        coalition = list(coalition)
        if len(coalition) == 0:
            continue
        if is_minimal_winning(coalition, utility, threshold):
            share = 1.0 / len(coalition)
            for i in coalition:
                dp_scores[i] += share

    if normalized:
        total = np.sum(dp_scores)
        if total > 0:
            dp_scores /= total

    return ValuationResult(
        algorithm="exact_deegan_packel_index",
        status="Converged",
        values=dp_scores,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )


def montecarlo_deegan_packel_index(utility, n_iterations=1000, threshold=None, normalized=True, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)
    players = np.arange(n)
    dp_scores = np.zeros(n)

    if threshold is None:
        threshold = utility(list(players)) / 2.0

    for _ in range(n_iterations):
        coalition_mask = np.random.choice([0, 1], size=n).astype(bool)
        coalition = np.where(coalition_mask)[0].tolist()
        if not coalition:
            continue

        if is_minimal_winning(coalition, utility, threshold):
            share = 1.0 / len(coalition)
            for i in coalition:
                dp_scores[i] += share

    if normalized:
        total = np.sum(dp_scores)
        if total > 0:
            dp_scores /= total

    return ValuationResult(
        algorithm="montecarlo_deegan_packel_index",
        status="Converged",
        values=dp_scores,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )
