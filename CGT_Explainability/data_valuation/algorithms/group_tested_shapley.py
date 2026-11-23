import numpy as np
import cvxpy as cp
from tqdm.auto import trange

class ValuationResult:
    def __init__(self, algorithm, status, values, data_names=None, solver_status=None):
        self.algorithm = algorithm
        self.status = status
        self.values = values
        self.data_names = data_names
        self.solver_status = solver_status


def _constants(n, epsilon, delta, utility_range):

    kk = np.arange(1, n)  # possible subset sizes
    Z = 2 * (1.0 / kk).sum()
    q = (1 / kk + 1 / (n - kk)) / Z
    q_tot = (n - 2) / n * q[0] + np.inner(q[1:], 1 + 2 * kk[1:] * (kk[1:] - n) / (n * (n - 1)))

    def h(u):
        return (1 + u) * np.log(1 + u) - u

    if utility_range == np.inf:
        T = -1
    else:
        T = 8 * np.log(n * (n - 1) / (2 * delta)) / (1 - q_tot**2)
        T /= h(2 * epsilon / (np.sqrt(n) * Z * utility_range * (1 - q_tot**2)))
    return kk, Z, q, q_tot, int(T)


def group_testing_shapley(utility_fn, n, n_samples, epsilon=0.1, delta=0.1,
                          utility_range=1.0, progress=False, solver=cp.SCS):

    kk, Z, q, q_tot, T_bound = _constants(n, epsilon, delta, utility_range)


    betas = np.zeros((n_samples, n), dtype=int)
    uu = np.empty(n_samples)

    rng = np.random.default_rng()

    # Step 1: Sample subsets and evaluate utility
    for t in trange(n_samples, disable=not progress):
        k = rng.choice(kk, p=q)  # choose subset size
        subset = rng.choice(np.arange(n), size=k, replace=False)
        uu[t] = utility_fn(subset)
        betas[t, subset] = 1

    # Step 2: Estimate pairwise differences
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            C[i, j] = np.dot(uu, betas[:, i] - betas[:, j])
    C *= Z / n_samples

    total_utility = utility_fn(np.arange(n))

    # Step 3: Solve feasibility problem with cvxpy
    v = cp.Variable(n)
    constraints = [cp.sum(v) == total_utility]
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(v[i] - v[j] <= epsilon + C[i, j])
            constraints.append(v[j] - v[i] <= epsilon - C[i, j])

    problem = cp.Problem(cp.Minimize(0), constraints)
    problem.solve(solver=solver)

    if problem.status != "optimal":
        values = np.nan * np.ones(n)
        status = "Failed"
    else:
        values = np.array(v.value).flatten()
        status = "Converged"

    return ValuationResult(
        algorithm="group_testing_shapley",
        status=status,
        values=values,
        solver_status=problem.status
    )
