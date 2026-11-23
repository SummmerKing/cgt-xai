import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from .least_core import ValuationResult

logger = logging.getLogger(__name__)

def leave_one_out(utility, progress: bool = False) -> ValuationResult:

    n = len(utility.data)
    values = np.zeros(n)

    all_indices = list(range(n))
    u_full = utility(all_indices)

    for i in range(n):
        subset = [j for j in range(n) if j != i]

        u_without_i = utility(subset)

        values[i] = u_full - u_without_i

    return ValuationResult(
        algorithm="leave_one_out",
        status="Converged",
        values=values,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names if hasattr(utility.data, 'data_names') else None
    )
