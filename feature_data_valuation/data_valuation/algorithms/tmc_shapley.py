import numpy as np
import logging
from typing import Optional
from tqdm.auto import tqdm
from .least_core import ValuationResult

logger = logging.getLogger(__name__)

def tmc_shapley(utility, n_iterations: int, tol: float = 1e-3, progress: bool = False) -> ValuationResult:


    n = len(utility.data)
    phi = np.zeros(n)

    full_utility = utility(list(range(n)))

    iterator = tqdm(range(1, n_iterations + 1)) if progress else range(1, n_iterations + 1)

    for t in iterator:
        permutation = np.random.permutation(n)

        prev_utility = utility([])

        for j, idx in enumerate(permutation, start=1):
            if abs(full_utility - prev_utility) < tol:
                current_utility = prev_utility
            else:
                subset = permutation[:j].tolist()
                current_utility = utility(subset)

            marginal = current_utility - prev_utility
            phi[idx] = (t - 1) / t * phi[idx] + (1 / t) * marginal

            prev_utility = current_utility

    return ValuationResult(
        algorithm="tmc_shapley",
        status="Converged",
        values=phi,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names if hasattr(utility.data, 'data_names') else None
    )
