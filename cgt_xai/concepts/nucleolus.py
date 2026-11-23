import logging
import numpy as np
import cvxpy as cp
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)
import itertools
from .least_core import ValuationResult

def powerset(iterable):
    from itertools import chain, combinations
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))

def exact_nucleolus(utility, epsilon: float = 0.0, options=None, progress: bool = False):

    if options is None:
        options = {}

    n = len(utility.data)


    coalitions = []
    for r in range(1, n + 1):
        for S in itertools.combinations(range(n), r):
            coalitions.append(list(S))

    m = len(coalitions)
    A = np.zeros((m, n))
    v = np.zeros(m)

    for i, S in enumerate(coalitions):
        A[i, S] = 1.0
        v[i] = utility(S)

    try:
        values, subsidies = _solve_nucleolus(A, v, epsilon=epsilon, **options)

        subsidy = subsidies[0] if subsidies is not None and len(subsidies) > 0 else None

        return ValuationResult(
            algorithm="exact_nucleolus",
            status="Converged",
            values=values,
            subsidy=subsidy,
            stderr=None,
            data_names=utility.data.data_names,
        )
    except Exception as e:
        logger.error(f"Error in exact nucleolus calculation: {e}")
        return ValuationResult(
            algorithm="exact_nucleolus",
            status="Failed",
            values=None,
            subsidy=None,
            stderr=str(e),
            data_names=utility.data.data_names,
        )


def montecarlo_nucleolus(utility, n_iterations, epsilon=0.0, options=None, progress=False):


    if options is None:
        options = {}

    n = len(utility.data)

    coalitions = []
    utility_values = []

    grand_coalition = list(range(n))
    coalitions.append(grand_coalition)
    utility_values.append(utility(grand_coalition))

    for _ in range(n_iterations - 1):
        subset_size = np.random.randint(1, n)
        subset = sorted(np.random.choice(n, subset_size, replace=False))

        utility_values.append(utility(subset))
        coalitions.append(subset)

    utility_values = np.array(utility_values)

    A = np.zeros((len(coalitions), n))
    for i, coalition in enumerate(coalitions):
        A[i, coalition] = 1

    try:
        values, subsidies = _solve_nucleolus(A, utility_values, epsilon=epsilon, **options)

        subsidy = subsidies[0]

        return ValuationResult(
            algorithm="montecarlo_nucleolus",
            status="Converged",
            values=values,
            subsidy=subsidy,
            stderr=None,
            data_names=utility.data.data_names,
        )
    except Exception as e:
        logger.error(f"Error in nucleolus calculation: {e}")
        return ValuationResult(
            algorithm="montecarlo_nucleolus",
            status="Failed",
            values=None,
            subsidy=None,
            stderr=None,
            data_names=utility.data.data_names,
        )

def _solve_nucleolus(A, v, epsilon=0.01, **options):

    n = A.shape[1]
    m = A.shape[0]

    grand_coalition_idx = np.argmax(np.sum(A, axis=1))

    active_coalitions = list(range(m))
    active_coalitions.remove(grand_coalition_idx)
    fixed_coalitions = [grand_coalition_idx]
    fixed_values = {}

    x = cp.Variable(n)
    excess = cp.Variable()

    constraints = [cp.sum(x) == v[grand_coalition_idx]]

    all_subsidies = np.zeros(m)

    while active_coalitions:

        constraints_this_round = constraints.copy()

        for i in active_coalitions:
            constraints_this_round.append(v[i] - cp.sum(cp.multiply(A[i], x)) <= excess)

        objective = cp.Minimize(excess)
        problem = cp.Problem(objective, constraints_this_round)

        try:
            problem.solve(**options)
        except Exception as e:
            logger.warning(f"Solver error: {e}")
            break

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"Problem not solved optimally: {problem.status}")
            break

        x_values = x.value
        max_excess = excess.value

        new_fixed = []
        for i in active_coalitions:
            current_excess = v[i] - np.dot(A[i], x_values)
            if abs(current_excess - max_excess) < 1e-5:
                new_fixed.append(i)
                fixed_values[i] = current_excess
                all_subsidies[i] = -current_excess

        if not new_fixed:
            logger.warning("No new coalitions fixed in this round")
            break

        for i in new_fixed:
            active_coalitions.remove(i)
            fixed_coalitions.append(i)

            constraints.append(v[i] - cp.sum(cp.multiply(A[i], x)) == fixed_values[i])

    nucleolus_values = x_values

    return nucleolus_values, all_subsidies

