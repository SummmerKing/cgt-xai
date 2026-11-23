import numpy as np
import math
import logging
from itertools import chain, combinations
from .least_core import ValuationResult

logger = logging.getLogger(__name__)

def powerset(iterable):
    """Generate all subsets of iterable."""
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))

def exact_banzhaf_index(utility, normalized=True, options=None, progress=False):
    """Compute exact Banzhaf index for all features."""
    if options is None:
        options = {}

    n = len(utility.data)
    banzhaf_scores = np.zeros(n)
    players = range(n)

    # For each player i
    for i in players:
        marginal_contributions = []

        # Iterate over all subsets that don't contain i
        for S in powerset([p for p in players if p != i]):
            S = list(S)

            # Compute marginal contribution: v(S∪{i}) - v(S)
            acc_with_feature = utility(S + [i])
            acc_without_feature = utility(S)
            marginal = acc_with_feature - acc_without_feature

            marginal_contributions.append(marginal)

        # Banzhaf value: average marginal contribution
        # Normalized by 2^(n-1) (number of subsets not containing i)
        banzhaf_scores[i] = sum(marginal_contributions) / (2 ** (n - 1))

    # Optional: Normalize to sum to 1
    if normalized:
        total = np.sum(banzhaf_scores)
        if total > 0:
            banzhaf_scores /= total
        else:
            logger.warning("Sum of Banzhaf scores is zero. Cannot normalize.")

    return ValuationResult(
        algorithm="exact_banzhaf_index",
        status="Converged",
        values=banzhaf_scores,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )


def montecarlo_banzhaf_index(utility, n_iterations=1000, normalized=True, options=None, progress=False):
    """Compute approximate Banzhaf index using Monte Carlo sampling."""
    if options is None:
        options = {}

    n = len(utility.data)
    banzhaf_scores = np.zeros(n)
    players = np.arange(n)

    for _ in range(n_iterations):
        # Randomly sample a coalition (each player included with prob 0.5)
        coalition = np.random.choice([0, 1], size=n).astype(bool)
        coalition_indices = np.where(coalition)[0].tolist()

        # For each player, compute marginal contribution
        for i in players:
            if i in coalition_indices:
                # If i is in the coalition, remove it
                without_i = [p for p in coalition_indices if p != i]
                marginal = utility(coalition_indices) - utility(without_i)
            else:
                # If i is not in the coalition, add it
                marginal = utility(coalition_indices + [i]) - utility(coalition_indices)

            banzhaf_scores[i] += marginal

    # Average over iterations
    banzhaf_scores /= n_iterations

    # Optional: Normalize to sum to 1
    if normalized:
        total = np.sum(banzhaf_scores)
        if total > 0:
            banzhaf_scores /= total
        else:
            logger.warning("Sum of Banzhaf scores is zero. Cannot normalize.")

    return ValuationResult(
        algorithm="montecarlo_banzhaf_index",
        status="Converged",
        values=banzhaf_scores,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )
