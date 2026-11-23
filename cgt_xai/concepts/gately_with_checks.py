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


    v_i = np.array([utility([i]) for i in players])


    v_N_without_i = np.array([utility([j for j in players if j != i]) for i in players])
    M_i = vN - v_N_without_i


    sum_v_i = np.sum(v_i)
    sum_M_i = np.sum(M_i)


    if not (sum_v_i <= vN + 1e-9 and vN <= sum_M_i + 1e-9):
        print(v_i)
        raise ValueError(
            f"Game is not Essential: sum(v_i)={sum_v_i:.4f} <= v(N)={vN:.4f} <= sum(M_i)={sum_M_i:.4f} is violated."
        )


    if np.any(v_i > M_i + 1e-9):
        violators = np.where(v_i > M_i + 1e-9)[0]
        raise ValueError(f"Game is not Semi-Standard: v({{i}}) > M_i(v) for players {violators}.")


    strict_standardness = np.any(v_i < M_i - 1e-9)
    if not strict_standardness:
        raise ValueError("Game is not Strictly Standard: v_j >= M_j(v) for all players, denominator would be zero.")


    denominator = np.sum(M_i - v_i)

    if np.isclose(denominator, 0.0):
         raise ValueError("Denominator is zero, even though strict standardness check passed.")

    adjustment = vN - sum_v_i

    numerator = M_i - v_i
    allocation = v_i + (numerator / denominator) * adjustment

    return ValuationResult(
        algorithm="exact_gately_value",
        status="Converged",
        values=allocation,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )


