import numpy as np
import logging
from .least_core import ValuationResult

logger = logging.getLogger(__name__)


import numpy as np
import logging
from .least_core import ValuationResult

logger = logging.getLogger(__name__)

def exact_gately_value(utility, options=None, progress=False):
    """Compute exact Gately point using the classical disruption formula."""
    if options is None:
        options = {}

    n = len(utility.data)
    players = list(range(n))
    vN = utility(players)

    singletons = np.array([utility([i]) for i in players])
    marginals = np.array([vN - utility([j for j in players if j != i]) for i in players])

    numerator = (n - 1) * vN - np.sum(vN - marginals)
    denominator = vN - np.sum(singletons)

    if np.isclose(denominator, 0.0):
        logger.warning("Denominator is zero. Returning equal split of grand coalition value.")
        allocation = np.full(n, vN / n)
        return ValuationResult(
            algorithm="exact_gately_value",
            status="Degenerate",
            values=allocation,
            subsidy=None,
            stderr=None,
            data_names=utility.data.data_names,
        )

    d_star = numerator / denominator

    if np.isclose(d_star, -1.0):
        logger.warning("d* ≈ -1, Gately point not unique. Returning centroid of imputation set.")
        surplus = (vN - np.sum(singletons)) / n
        allocation = singletons + surplus
    else:
        allocation = (marginals + d_star * singletons) / (d_star + 1)

    return ValuationResult(
        algorithm="exact_gately_value",
        status="Converged",
        values=allocation,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )
