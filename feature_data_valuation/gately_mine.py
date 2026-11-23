import numpy as np

def gately_point(v, N=None, verbose=True):
    """Compute the exact Gately point for a transferable-utility (TU) game."""
    if N is None:
        N = sorted({i for coalition in v.keys() for i in coalition})
    n = len(N)

    vN = v[tuple(N)]                      # grand coalition value
    singletons = np.array([v[(i,)] for i in N])
    marginals = np.array([vN - v[tuple(sorted(set(N) - {i}))] for i in N])

    numerator = (n - 1) * vN - np.sum(vN - marginals)
    denominator = vN - np.sum(singletons)
    if np.isclose(denominator, 0.0):
        raise ValueError("Degenerate case: v(N) = sum(singletons)")

    d_star = numerator / denominator

    allocation = (marginals + d_star * singletons) / (1 + d_star)

    if verbose:
        print("Players:", N)
        print("v(N) =", vN)
        print("Singletons v({i}) =", singletons)
        print("Marginals v(N)-v(N\\{i}) =", marginals)
        print("d* =", d_star)
        print("Gately allocation =", allocation)
        print("Sum check =", np.sum(allocation))

    return allocation, d_star


v = {
    (1,): 3,
    (2,): 4,
    (3,): 5,
    (1,2): 9,
    (1,3): 10,
    (2,3): 11,
    (1,2,3): 14.5
}

alloc, d_star = gately_point(v)

expected = np.array([18/11, 36/11, 12/11])
print("\nExpected (from book) =", expected)
print("Difference =", alloc - expected)
