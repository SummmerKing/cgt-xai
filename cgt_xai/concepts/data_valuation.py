import numpy as np
from tqdm.auto import tqdm
from .least_core import ValuationResult

def naive_loo(utility, progress=False):
    """Leave-one-out data valuation.
    
        Computes the difference in model performance when each data point is removed.
    """
    n_samples = len(utility.data.indices)
    values = np.zeros(n_samples)

    # Get performance with all samples
    full_perf = utility(utility.data.indices)

    # Get performance without each sample
    iterator = tqdm(range(n_samples), desc="LOO Valuation") if progress else range(n_samples)
    for i in iterator:
        subset = [j for j in utility.data.indices if j != i]
        values[i] = full_perf - utility(subset)

    return ValuationResult(
        algorithm="leave_one_out",
        status="Converged",
        values=values,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names
    )

def random_valuation(utility, seed=42):
    """Random baseline valuation.
    
        Assigns random values to each data point.
    """
    n_samples = len(utility.data.indices)
    np.random.seed(seed)
    values = np.random.rand(n_samples)

    return ValuationResult(
        algorithm="random",
        status="Converged",
        values=values,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names
    )

def tmc_shapley(utility, n_iterations=100, seed=42, progress=False):
    """Truncated Monte Carlo Shapley approximation.
    
        Approximates Shapley values by sampling permutations of data points.
    """
    n_samples = len(utility.data.indices)
    values = np.zeros(n_samples)
    rng = np.random.RandomState(seed)

    iterator = tqdm(range(n_iterations), desc="TMC Shapley") if progress else range(n_iterations)
    for _ in iterator:
        # Create a random permutation of data points
        perm = rng.permutation(n_samples)
        marginal_contributions = np.zeros(n_samples)

        # Initialize with empty set
        prev_value = utility([])

        # Compute marginal contributions along this permutation
        for i, idx in enumerate(perm):
            curr_subset = perm[:i+1].tolist()
            curr_value = utility(curr_subset)
            marginal_contributions[perm[i]] = curr_value - prev_value
            prev_value = curr_value

        values += marginal_contributions

    values /= n_iterations
    return ValuationResult(
        algorithm="tmc_shapley",
        status="Converged",
        values=values,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names
    )

def group_testing_shapley(utility, n_iterations=100, seed=42, progress=False):
    """Group Testing Shapley approximation.
    
        Estimates Shapley values more efficiently by sampling random subset sizes.
    """
    n_samples = len(utility.data.indices)
    values = np.zeros(n_samples)
    counts = np.zeros(n_samples)
    rng = np.random.RandomState(seed)

    iterator = tqdm(range(n_iterations), desc="Group Testing Shapley") if progress else range(n_iterations)
    for _ in iterator:
        # Random subset size
        subset_size = rng.randint(1, n_samples)
        # Random subset
        subset = rng.choice(n_samples, size=subset_size, replace=False)

        # Value with the subset
        val_with = utility(subset)

        # For each element in the subset, compute value without it
        for idx in subset:
            subset_without = [s for s in subset if s != idx]
            val_without = utility(subset_without)

            # Marginal contribution
            values[idx] += val_with - val_without
            counts[idx] += 1

    # Average the values
    for i in range(n_samples):
        if counts[i] > 0:
            values[i] /= counts[i]

    return ValuationResult(
        algorithm="group_testing_shapley",
        status="Converged",
        values=values,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names
    )
