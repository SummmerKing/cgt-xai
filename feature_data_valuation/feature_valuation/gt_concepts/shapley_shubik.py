import numpy as np
import math
import logging
import itertools
from .least_core import ValuationResult

logger = logging.getLogger(__name__)


def exact_shapley_shubik(utility, accuracy_threshold):
    n = len(utility.data)
    scores = np.zeros(n)
    total_perms = math.factorial(n)
    players = list(range(n))

    for perm in itertools.permutations(players):
        coalition = []
        # Coalition loses initially by definition (empty coalition utility < threshold)
        # If empty utility already wins, no feature is pivotal
        prev_value = utility(coalition)
        if prev_value >= accuracy_threshold:
            continue  # No pivot in this permutation

        for p in perm:
            coalition.append(p)
            current_value = utility(coalition)

            # PIVOTAL CONDITION: losing -> winning
            if prev_value < accuracy_threshold and current_value >= accuracy_threshold:
                scores[p] += 1
                break  # Only the first pivot counts per permutation

            prev_value = current_value

    scores /= total_perms
    return ValuationResult(
        algorithm="exact_shapley_shubik",
        status="Converged",
        values=scores,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )


def montecarlo_shapley_shubik(utility, accuracy_threshold, n_iterations=1000, options=None, progress=False):

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

            if np.isnan(current_value) or np.isnan(prev_value):

                continue


            if prev_value < accuracy_threshold <= current_value:
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
