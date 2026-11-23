import numpy as np
import itertools
from typing import List
import logging
import math
from .least_core import ValuationResult
from itertools import chain, combinations

logger = logging.getLogger(__name__)

def exact_shapley_value(utility, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)
    shapley_values = np.zeros(n)
    factorial_n = math.factorial(n)

    players = range(n)

    for i in players:
        contribution = 0.0
        for S in powerset([p for p in players if p != i]):
            S = list(S)
            weight = (math.factorial(len(S)) *
                      math.factorial(n - len(S) - 1)) / factorial_n
            marginal = utility(S + [i]) - utility(S)
            contribution += weight * marginal
        shapley_values[i] = contribution

    return ValuationResult(
        algorithm="exact_shapley_value",
        status="Converged",
        values=shapley_values,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )

def powerset(iterable):


    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))

def montecarlo_shapley_value(utility, n_iterations=1000, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)
    shapley_values = np.zeros(n)

    players = np.arange(n)

    for _ in range(n_iterations):
        perm = np.random.permutation(players)
        current_coalition = []
        prev_value = utility(current_coalition)

        for p in perm:
            current_value = utility(current_coalition + [p])
            marginal = current_value - prev_value
            shapley_values[p] += marginal
            current_coalition.append(p)
            prev_value = current_value

    shapley_values /= n_iterations

    return ValuationResult(
        algorithm="montecarlo_shapley_value",
        status="Converged",
        values=shapley_values,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )


