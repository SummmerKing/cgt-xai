#!/usr/bin/env python3
"""Faithfulness and robustness evaluation of game-theoretic feature attributions on tabular datasets."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, List
from pathlib import Path
import inspect

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Import project modules at top (robust root detection)
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = Path.cwd().resolve()

root = None
for base in [SCRIPT_DIR, HERE]:
    for p in [base] + list(base.parents):
        if (p / "models").is_dir() and (p / "gt_concepts").is_dir():
            root = p
            break
    if root is not None:
        break

if root is None:
    raise RuntimeError(
        f"Could not find project root from script_dir={SCRIPT_DIR} or cwd={HERE}. "
        f"Ensure folders 'models/' and 'gt_concepts/' exist in the project root."
    )

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

print("Project root added:", root)

from models.train_model import get_model  # noqa: E402
from gt_concepts.gately_point import exact_gately_value  # noqa: E402
from gt_concepts.shapley import montecarlo_shapley_value  # noqa: E402
from gt_concepts.least_core import montecarlo_least_core  # noqa: E402

_SHAPLEY_SIG = inspect.signature(montecarlo_shapley_value)
_LEASTCORE_SIG = inspect.signature(montecarlo_least_core)

class DataWrapper:
    def __init__(self, indices, names=None):
        self.indices = indices
        self.data_names = names if names is not None else [f"f{i}" for i in range(len(indices))]

    def __len__(self):
        return len(self.indices)


class Utility:
    """v(S): train model on features S, evaluate on test."""
    def __init__(self, X_train, y_train, X_test, y_test, model_name, scorer,
                 feature_names=None, random_state=42):
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)
        self.model_name = model_name
        self.scorer = scorer
        self.data = DataWrapper(list(range(self.X_train.shape[1])), feature_names)
        self.random_state = random_state
        self._cache = {}

    def __call__(self, subset):
        key = tuple(sorted(subset))
        if key in self._cache:
            return self._cache[key]

        # Empty coalition baseline
        if len(key) == 0:
            majority = np.bincount(self.y_train.astype(int)).argmax()
            y_pred = np.full_like(self.y_test, majority)
            score = float(self.scorer(self.y_test, y_pred))
            if not np.isfinite(score):
                score = 0.0
            self._cache[key] = score
            return score

        model = get_model(self.model_name, self.random_state)
        model.fit(self.X_train[:, key], self.y_train)
        y_pred = model.predict(self.X_test[:, key])
        score = float(self.scorer(self.y_test, y_pred))
        if not np.isfinite(score):
            score = 0.0
        self._cache[key] = score
        return score


# Attribution methods with best-effort determinism
def gately_attribution(utility: Utility) -> Dict[str, float]:
    """Gately (exact, deterministic by design)."""
    res = exact_gately_value(utility, options=None, progress=False)
    return {utility.data.data_names[i]: float(res.values[i]) for i in range(len(res.values))}


def mc_shapley_attribution(utility: Utility, n_iterations: int = 500) -> Dict[str, float]:
    """MC Shapley with best-effort determinism."""
    np.random.seed(utility.random_state)
    sig = _SHAPLEY_SIG

    try:
        if 'random_state' in sig.parameters:
            res = montecarlo_shapley_value(
                utility,
                n_iterations=n_iterations,
                random_state=utility.random_state,
                progress=False
            )
        else:
            res = montecarlo_shapley_value(
                utility,
                n_iterations=n_iterations,
                options={"random_state": utility.random_state},
                progress=False
            )
    except (TypeError, KeyError):
        res = montecarlo_shapley_value(
            utility,
            n_iterations=n_iterations,
            options=None,
            progress=False
        )

    return {utility.data.data_names[i]: float(res.values[i]) for i in range(len(res.values))}


def least_core_attribution(utility: Utility, n_iterations: int = 500) -> Dict[str, float]:
    """Least Core with best-effort determinism (no changes to least_core.py)."""
    np.random.seed(utility.random_state)
    sig = _LEASTCORE_SIG
    n = len(utility.data)

    # Minimal, solver-safe options (avoid passing wrong kwargs)
    # CLARABEL is often more stable than SCS
    solver_chain = [
        ("CLARABEL", {"solver": "CLARABEL", "verbose": False}),
        ("SCIPY",    {"solver": "SCIPY", "verbose": False}),
        ("SCS",      {"solver": "SCS", "verbose": False}),
    ]

    last_err = None

    for solver_name, opts in solver_chain:
        try:
            if 'random_state' in sig.parameters:
                res = montecarlo_least_core(
                    utility,
                    n_iterations=n_iterations,
                    epsilon=1e-6,
                    random_state=utility.random_state,
                    options=opts,
                    progress=False
                )
            else:
                # Pass random_state via options if supported upstream; safe if ignored
                o = dict(opts)
                o["random_state"] = utility.random_state
                res = montecarlo_least_core(
                    utility,
                    n_iterations=n_iterations,
                    epsilon=1e-6,
                    options=o,
                    progress=False
                )

            vals = getattr(res, "values", None)
            if vals is None:
                # solver returned non-optimal -> least_core.py returns None,None -> values None
                raise RuntimeError(f"{solver_name} returned None values")

            vals = np.asarray(vals, dtype=float)
            if vals.shape[0] != n or not np.isfinite(vals).all():
                raise RuntimeError(f"{solver_name} produced invalid values")

            return {utility.data.data_names[i]: float(vals[i]) for i in range(n)}

        except Exception as e:
            last_err = e
            # continue trying next solver
            continue

    print(f"[WARN] least_core failed for all available solvers. seed={utility.random_state}. Last error: {type(last_err).__name__}: {last_err}")
    return {utility.data.data_names[i]: 0.0 for i in range(n)}


# Data loading (unchanged)
def load_mi_complications(binary_let_is=True, random_state=42):
    """Load MI complications dataset."""
    path = "../datasets/MI.data"
    df = pd.read_csv(path, header=None, na_values="?")

    X_df = df.iloc[:, 1:112]
    feature_names = [f"feat_{i}" for i in range(1, X_df.shape[1] + 1)]

    let_is = df.iloc[:, 123].astype(int).values
    y = (let_is != 0).astype(int) if binary_let_is else let_is

    imputer = SimpleImputer(strategy="most_frequent")
    X = imputer.fit_transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names


# Permutation importance weights
def compute_permutation_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model_name: str = "svm",
    scorer=accuracy_score,
    n_repeats: int = 50,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    model = get_model(model_name, seed)
    model.fit(X_train, y_train)

    y_pred_base = model.predict(X_test)
    baseline_score = scorer(y_test, y_pred_base)

    n_features = len(feature_names)
    drops = np.zeros(n_features, dtype=float)

    for feat_idx in range(n_features):
        drop_list = []
        for _ in range(n_repeats):
            X_test_perm = X_test.copy()
            X_test_perm[:, feat_idx] = rng.permutation(X_test_perm[:, feat_idx])

            y_pred_perm = model.predict(X_test_perm)
            perm_score = scorer(y_test, y_pred_perm)

            drop_list.append(baseline_score - perm_score)

        drops[feat_idx] = max(0.0, float(np.mean(drop_list)))

    zero_frac = float((drops <= 1e-12).sum() / n_features)
    if verbose:
        print(f"[permutation weights] {zero_frac:.1%} of features have zero weight")
        print(f"[permutation weights] baseline {scorer.__name__} = {baseline_score:.4f}")

    if drops.sum() <= 1e-12:
        if verbose:
            print("[permutation weights] WARNING: all drops zero, using uniform")
        return {feature_names[i]: 1.0 / n_features for i in range(n_features)}

    w = drops / drops.sum()
    return {feature_names[i]: float(w[i]) for i in range(n_features)}


# Weighted Shannon entropy
def weighted_rank_entropy(
    rank_counts: np.ndarray,
    rank_position_weights: np.ndarray,
    eps: float = 1e-12,
) -> float:
    total = rank_counts.sum()
    if total <= 0:
        return 0.0

    p = rank_counts / total
    w_pos = rank_position_weights / (rank_position_weights.sum() + eps)
    H = -float(np.sum(w_pos * p * np.log2(p + eps)))
    return H


@dataclass
class PrimeResult:
    prime_total: float
    prime_normalized: float
    per_feature_entropy: Dict[str, float]
    per_feature_entropy_normalized: Dict[str, float]
    perm_importance_weights: Dict[str, float]
    rank_position_weights: np.ndarray
    n_rankings: int
    zero_weight_fraction: float


# PRIME: Main computation
def compute_prime(
    attribution_fn: Callable[[Utility], Dict[str, float]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model_name: str = "svm",
    scorer=accuracy_score,
    num_permutations: int = 20,
    mc_iterations: int = 500,
    seed: int = 42,
    rank_weight_scheme: str = "harmonic",
    verbose: bool = False,
) -> PrimeResult:
    rng = np.random.default_rng(seed)
    p = len(feature_names)

    if rank_weight_scheme == "harmonic":
        w_pos = 1.0 / (np.arange(p, dtype=float) + 1.0)
    elif rank_weight_scheme == "uniform":
        w_pos = np.ones(p, dtype=float)
    elif rank_weight_scheme == "exponential":
        w_pos = np.exp(-0.25 * np.arange(p, dtype=float))
    else:
        raise ValueError("rank_weight_scheme must be 'harmonic', 'uniform', or 'exponential'")

    if verbose:
        print("Computing test-set permutation importance weights w_i...")

    w_feat = compute_permutation_weights(
        X_train, y_train, X_test, y_test,
        feature_names=feature_names,
        model_name=model_name,
        scorer=scorer,
        n_repeats=50,
        seed=seed + 1000,
        verbose=verbose,
    )

    zero_weight_frac = float(sum(1 for w in w_feat.values() if w <= 1e-12) / p)

    if verbose:
        top5 = sorted(w_feat.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top-5 features by w_i: {[(f, f'{w:.4f}') for f, w in top5]}")

    sig = inspect.signature(attribution_fn)
    supports_n_iter = 'n_iterations' in sig.parameters

    rank_counts = {f: np.zeros(p, dtype=float) for f in feature_names}
    n_rankings = 0

    if verbose:
        print(f"\nMeasuring ranking stability under training perturbations...")
        print(f"  {num_permutations} perms × {p} features = {num_permutations * p} rankings")

    for t in range(num_permutations):
        for feat_idx, permuted_feature in enumerate(feature_names):
            X_train_pert = X_train.copy()
            X_train_pert[:, feat_idx] = rng.permutation(X_train_pert[:, feat_idx])

            utility_pert = Utility(
                X_train_pert, y_train,
                X_test, y_test,
                model_name=model_name,
                scorer=scorer,
                feature_names=feature_names,
                random_state=seed + t * p + feat_idx
            )

            if supports_n_iter:
                vals = attribution_fn(utility_pert, n_iterations=mc_iterations)
            else:
                vals = attribution_fn(utility_pert)

            # Safe ranking (handles None or non-finite)
            def safe_abs(v):
                try:
                    vv = float(v)
                    if not np.isfinite(vv):
                        return 0.0
                    return abs(vv)
                except Exception:
                    return 0.0

            ranking = sorted(feature_names, key=lambda f: safe_abs(vals.get(f, 0.0)), reverse=True)

            for pos, feat in enumerate(ranking):
                rank_counts[feat][pos] += 1.0

            n_rankings += 1

            if verbose and t == 0 and feat_idx < 2:
                print(f"  [sample] permuted={permuted_feature}, top5={ranking[:5]}")

    H = {}
    H_norm = {}
    for feat in feature_names:
        H_feat = weighted_rank_entropy(rank_counts[feat], w_pos)
        H[feat] = H_feat
        H_norm[feat] = H_feat / (np.log2(p) + 1e-12)

    if verbose:
        print(f"\nMost unstable features (highest entropy):")
        for f in sorted(feature_names, key=lambda x: H[x], reverse=True)[:5]:
            print(f"  {f}: H={H[f]:.4f}, H_norm={H_norm[f]:.4f}")

    prime_total = float(sum(w_feat[f] * H[f] for f in feature_names))
    prime_normalized = prime_total / (np.log2(p) + 1e-12)

    if verbose:
        print(f"\nPRIME Results:")
        print(f"  PRIME_total = {prime_total:.4f}")
        print(f"  PRIME_normalized = {prime_normalized:.4f} (lower = more consistent)")
        print(f"  Zero-weight features: {zero_weight_frac:.1%}")
        print(f"  Total rankings collected: {n_rankings}")

    return PrimeResult(
        prime_total=prime_total,
        prime_normalized=prime_normalized,
        per_feature_entropy=H,
        per_feature_entropy_normalized=H_norm,
        perm_importance_weights=w_feat,
        rank_position_weights=w_pos / (w_pos.sum() + 1e-12),
        n_rankings=n_rankings,
        zero_weight_fraction=zero_weight_frac,
    )


# Determinism verification helper
def verify_determinism(
    attribution_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model_name: str = "svm",
    seed: int = 42,
    det_mc_iterations: int = 25,
    n_trials: int = 3,
) -> bool:
    print(f"Verifying determinism for {attribution_fn.__name__}...")

    results = []
    sig = inspect.signature(attribution_fn)
    supports_n_iter = 'n_iterations' in sig.parameters

    for _ in range(n_trials):
        utility = Utility(
            X_train, y_train, X_test, y_test,
            model_name=model_name,
            scorer=accuracy_score,
            feature_names=feature_names,
            random_state=seed
        )

        if supports_n_iter:
            vals = attribution_fn(utility, n_iterations=det_mc_iterations)
        else:
            vals = attribution_fn(utility)

        results.append(vals)

    first = results[0]
    all_match = True
    for r in results[1:]:
        for f in feature_names:
            if abs(r.get(f, 0.0) - first.get(f, 0.0)) >= 1e-10:
                all_match = False
                break
        if not all_match:
            break

    if all_match:
        print(f"  Deterministic: {n_trials} trials produced identical results")
    else:
        print(f"  Error: NON-deterministic: results differ across trials")
        max_diff = 0.0
        for r in results[1:]:
            max_diff = max(
                max_diff,
                max(abs(r.get(f, 0.0) - first.get(f, 0.0)) for f in feature_names)
            )
        print(f"     Max difference (worst over trials): {max_diff:.6e}")

    return all_match


# Run PRIME for all methods
def run_prime_mi_complications(
    model_name: str = "svm",
    num_permutations: int = 20,
    mc_iterations: int = 500,
    seed: int = 42,
    verify_determinism_first: bool = True,
    verbose: bool = True,
):
    print("="*70)
    print("PRIME Evaluation: mi_complications")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"MC iterations: {mc_iterations}")
    print(f"Training perturbations: {num_permutations}")
    print(f"Weight computation: test-set permutation importance (accuracy)")
    print(f"Stability measure: ranking entropy under training noise")
    print("="*70)

    X_train, X_test, y_train, y_test, feature_names = load_mi_complications(
        binary_let_is=True,
        random_state=seed
    )

    print(f"\nDataset: n_train={X_train.shape[0]}, n_test={X_test.shape[0]}, n_features={len(feature_names)}")

    attribution_methods = {
        "least_core_mc": least_core_attribution,
        "mc_shapley": mc_shapley_attribution,
    }

    if verify_determinism_first:
        print(f"\n{'='*70}")
        print("Determinism Verification")
        print("="*70)
        for method_name, attrib_fn in attribution_methods.items():
            verify_determinism(
                attrib_fn, X_train, y_train, X_test, y_test,
                feature_names, model_name, seed, det_mc_iterations=25, n_trials=3
            )
        print()

    results = []
    for method_name, attrib_fn in attribution_methods.items():
        print(f"\n{'='*70}")
        print(f"Method: {method_name}")
        print(f"{'='*70}")

        prime_res = compute_prime(
            attribution_fn=attrib_fn,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            model_name=model_name,
            scorer=accuracy_score,
            num_permutations=num_permutations,
            mc_iterations=mc_iterations,
            seed=seed,
            rank_weight_scheme="harmonic",
            verbose=verbose,
        )

        results.append({
            "dataset": "mi_complications",
            "model": model_name,
            "method": method_name,
            "PRIME_total": prime_res.prime_total,
            "PRIME_normalized": prime_res.prime_normalized,
            "zero_weight_pct": prime_res.zero_weight_fraction * 100,
            "n_rankings": prime_res.n_rankings,
            "mc_iterations": mc_iterations,
        })

    df_results = pd.DataFrame(results).sort_values("PRIME_normalized")

    print(f"\n{'='*70}")
    print("PRIME Results Summary")
    print("="*70)
    print("Lower PRIME = more consistent rankings under training perturbations")
    print("="*70)
    print(df_results.to_string(index=False))

    return df_results


# Main
def main():
    df_prime = run_prime_mi_complications(
        model_name="svm",
        num_permutations=5,
        mc_iterations=100,
        seed=42,
        verify_determinism_first=False,
        verbose=True,
    )

    output_path = Path("./output/PRIME_results_mi_complications_final.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_prime.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
