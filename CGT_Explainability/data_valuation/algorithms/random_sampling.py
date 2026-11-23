import numpy as np
import logging
from .least_core import ValuationResult

logger = logging.getLogger(__name__)

def random_valuation(utility, random_state=None) -> ValuationResult:

    n = len(utility.data)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random values
    values = np.random.rand(n)

    return ValuationResult(
        algorithm="random",
        status="Converged",
        values=values,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names if hasattr(utility.data, 'data_names') else None
    )
