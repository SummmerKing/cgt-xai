import numpy as np
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def exact_least_core(utility, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)

    all_subsets = list(powerset(range(n)))
    utility_values = np.zeros(len(all_subsets))
    constraints = np.zeros((len(all_subsets), n))

    for i, subset in enumerate(all_subsets):
        utility_values[i] = utility(subset)
        for j in subset:
            constraints[i, j] = 1

    constraints = constraints[1:]
    utility_values = utility_values[1:]

    values, subsidy = _solve_least_core_linear_program(
        A_eq=np.ones((1, n)),
        b_eq=np.array([utility(list(range(n)))]),
        A_lb=constraints,
        b_lb=utility_values,
        **options
    )

    return ValuationResult(
        algorithm="exact_least_core",
        status="Converged",
        values=values,
        subsidy=subsidy,
        stderr=None,
        data_names=utility.data.data_names,
    )

def montecarlo_least_core(utility, n_iterations, epsilon=0.0, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)

    utility_values = []
    constraints = []

    for _ in range(n_iterations):
        subset_size = np.random.randint(1, n)
        subset = sorted(np.random.choice(n, subset_size, replace=False))


        utility_val = utility(subset)
        utility_values.append(utility_val)


        constraint = np.zeros(n)
        constraint[subset] = 1
        constraints.append(constraint)

    utility_values = np.array(utility_values)
    constraints = np.array(constraints)


    values, subsidy = _solve_least_core_linear_program(
        A_eq=np.ones((1, n)),
        b_eq=np.array([utility(list(range(n)))]),
        A_lb=constraints,
        b_lb=utility_values,
        epsilon=epsilon,
        **options
    )

    return ValuationResult(
        algorithm="montecarlo_least_core",
        status="Converged",
        values=values,
        subsidy=subsidy,
        stderr=None,
        data_names=utility.data.data_names,
    )
def _solve_egalitarian_least_core_quadratic_program(subsidy, A_eq, b_eq, A_lb, b_lb, epsilon=0.0, **options):

    n = A_eq.shape[1]

    x = cp.Variable(n)

    excesses = b_lb - A_lb @ x - subsidy
    objective = cp.Minimize(cp.sum_squares(excesses))

    constraints = [
        A_eq @ x == b_eq,
        A_lb @ x + subsidy >= b_lb - epsilon
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(**options)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return None

    return x.value

def _solve_least_core_linear_program(A_eq, b_eq, A_lb, b_lb, epsilon=0.0, **options):

    n = A_eq.shape[1]

    x = cp.Variable(n)
    subsidy = cp.Variable()

    objective = cp.Minimize(subsidy)

    constraints = [
        A_eq @ x == b_eq,
        A_lb @ x + subsidy >= b_lb - epsilon
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(**options)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return None, None

    return x.value, subsidy.value

def powerset(indices):
    from itertools import chain, combinations
    return chain.from_iterable(combinations(indices, r) for r in range(len(indices)+1))

class ValuationResult:

    def __init__(self, algorithm, status, values, subsidy, stderr, data_names):
        self.algorithm = algorithm
        self.status = status
        self.values = values
        self.subsidy = subsidy
        self.stderr = stderr
        self.data_names = data_names


