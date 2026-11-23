import numpy as np
import logging
from .least_core import ValuationResult

logger = logging.getLogger(__name__)

def exact_gately_value(utility, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)
    players = list(range(n))

    vN = utility(players)
    singletons = np.array([utility([i]) for i in players])
    utopia = np.array([vN - utility([j for j in players if j != i]) for i in players])

    denom = np.sum(utopia - singletons)

    if np.isclose(denom, 0.0):
        logger.warning("EPD = -1, Gately value not unique. Returning centroid of imputation set.")
        surplus = (vN - np.sum(singletons)) / n
        allocation = singletons + surplus
    else:
        lam = (vN - np.sum(singletons)) / denom
        allocation = singletons + lam * (utopia - singletons)

    return ValuationResult(
        algorithm="exact_gately_value",
        status="Converged",
        values=allocation,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )


def montecarlo_gately_value(utility, n_iterations=1000, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)
    players = list(range(n))

    vN = utility(players)
    singletons = np.zeros(n)
    utopia = np.zeros(n)

    for _ in range(n_iterations):

        i = np.random.randint(0, n)

        singletons[i] += utility([i])

        coalition = [j for j in players if j != i]
        sampled = np.random.choice(coalition, size=len(coalition)//2, replace=False).tolist()
        utopia[i] += vN - utility(sampled)

    singletons /= n_iterations
    utopia /= n_iterations

    denom = np.sum(utopia - singletons)

    if np.isclose(denom, 0.0):
        logger.warning("EPD ≈ -1, Monte Carlo Gately value not unique. Using centroid of imputation set.")
        surplus = (vN - np.sum(singletons)) / n
        allocation = singletons + surplus
    else:
        lam = (vN - np.sum(singletons)) / denom
        allocation = singletons + lam * (utopia - singletons)

    return ValuationResult(
        algorithm="montecarlo_gately_value",
        status="Converged",
        values=allocation,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )
