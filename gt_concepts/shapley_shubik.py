import numpy as np
import math
import logging
import itertools
from .least_core import ValuationResult

logger = logging.getLogger(__name__)

def exact_shapley_shubik(utility, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)
    players = list(range(n))
    shapley_shubik_scores = np.zeros(n)
    total_perms = math.factorial(n)

    for perm in itertools.permutations(players):
        current_coalition = []
        prev_value = utility(current_coalition)
        for idx, p in enumerate(perm):
            current_value = utility(current_coalition + [p])
            if current_value > prev_value:
                shapley_shubik_scores[p] += 1
                break
            current_coalition.append(p)
            prev_value = current_value

    shapley_shubik_scores /= total_perms

    return ValuationResult(
        algorithm="exact_shapley_shubik",
        status="Converged",
        values=shapley_shubik_scores,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )


def montecarlo_shapley_shubik(utility, n_iterations=1000, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)
    players = np.arange(n)
    shapley_shubik_scores = np.zeros(n)

    for _ in range(n_iterations):
        perm = np.random.permutation(players)
        current_coalition = []
        prev_value = utility(current_coalition)

        for p in perm:
            current_value = utility(current_coalition + [p])
            if current_value > prev_value:
                shapley_shubik_scores[p] += 1
                break
            current_coalition.append(p)
            prev_value = current_value

    shapley_shubik_scores /= n_iterations

    return ValuationResult(
        algorithm="montecarlo_shapley_shubik",
        status="Converged",
        values=shapley_shubik_scores,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )
