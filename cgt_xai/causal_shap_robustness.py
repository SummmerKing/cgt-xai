"""Robustness analysis for Causal SHAP attributions.

Baseline and perturbed runs share the same test instances, permutation orders and SEM noise draws,
so differences come only from the perturbation.
"""

import re
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple, Set
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 130})


# Config

ROBUSTNESS_CONFIG = {
    "epsilons":    [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00],

    "n_trials":    10,   # independent perturbation draws per epsilon (for error bars)
    "k_instances": 50,   # test instances per global SHAP estimate
    "n_perms":     100,  # Shapley permutations per instance  <- biggest noise reducer
    "M_per_sem":   32,   # interventional MC draws per SEM per coalition
    "n_dags":      5,    # DAGs sampled from CPDAG
    "alpha_pc":    0.1,  # PC algorithm significance level
    "random_seed": 42,
    "test_size":   0.25,
}


COMMON_TARGETS = [
    "target", "label", "y", "class", "outcome", "cvd", "cardio",
    "TenYearCHD", "heart_disease", "stroke", "risk", "cancer",
    "lung_cancer", "has_cancer", "diagnosis", "cv_risk", "lung_cancer_risk",
]


# 1. data helpers

def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if not re.match(r"^unnamed", c, re.I)]].copy()


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    for c in df.columns:
        if c not in num.columns:
            try:
                num[c] = pd.to_numeric(df[c], errors="raise")
            except Exception:
                pass
    return num


def _detect_target(df: pd.DataFrame, hint: Optional[str] = None) -> str:
    if hint and hint in df.columns:
        return hint
    lmap = {c.lower(): c for c in df.columns}
    for t in COMMON_TARGETS:
        if t.lower() in lmap:
            return lmap[t.lower()]
    return df.columns[-1]


# 2. perturbable sem

class PerturbableSEM:
    """Linear Gaussian SEM fitted on one DAG."""

    def __init__(self, dag: nx.DiGraph, feature_names: List[str]):
        self.dag = dag
        self.features = list(feature_names)
        self.base_mechs: Dict[str, dict] = {}
        self.topo: List[str] = []

    def fit(self, X: pd.DataFrame) -> "PerturbableSEM":
        try:
            self.topo = list(nx.topological_sort(self.dag))
        except Exception:
            self.topo = self.features[:]

        for var in self.features:
            parents = list(self.dag.predecessors(var))
            if not parents:
                self.base_mechs[var] = {
                    "parents":   [],
                    "intercept": float(X[var].mean()),
                    "coef":      np.array([]),
                    "sigma":     float(X[var].std(ddof=0)),
                }
            else:
                lr = LinearRegression().fit(X[parents].values, X[var].values)
                resid = X[var].values - lr.predict(X[parents].values)
                self.base_mechs[var] = {
                    "parents":   parents,
                    "intercept": float(lr.intercept_),
                    "coef":      lr.coef_.astype(float).copy(),
                    "sigma":     float(np.std(resid, ddof=0)),
                }
        return self

    def perturbed_mechs(
        self, epsilon: float, rng: np.random.Generator
    ) -> Dict[str, dict]:
        """Return a deep copy of mechanisms with edge weights perturbed by epsilon."""
        mechs = {}
        for var, m in self.base_mechs.items():
            mc = copy.deepcopy(m)
            if mc["coef"].size > 0 and epsilon > 0:
                delta = rng.uniform(-1.0, 1.0, size=mc["coef"].shape)
                mc["coef"] = mc["coef"] * (1.0 + epsilon * delta)
            mechs[var] = mc
        return mechs

    def sample_intervention(
        self,
        S: Set[str],
        x_obs: pd.Series,
        mechs: Dict[str, dict],
        noise_rng: np.random.Generator,
    ) -> Dict[str, float]:
        """Sample one interventional world do(X_S = x_S) for coalition S."""
        x_new: Dict[str, float] = {}
        for var in self.topo:
            if var in S:
                x_new[var] = float(x_obs[var])
                continue
            m = mechs[var]
            if not m["parents"]:
                mu, sig = m["intercept"], m["sigma"]
                x_new[var] = mu if sig < 1e-9 else float(noise_rng.normal(mu, sig))
            else:
                pv = np.array([x_new.get(p, float(x_obs[p])) for p in m["parents"]])
                mu = m["intercept"] + float(pv @ m["coef"])
                sig = m["sigma"]
                x_new[var] = mu if sig < 1e-9 else float(noise_rng.normal(mu, sig))
        return x_new


# 3. prediction helpers

def _predict_scalar(model, X_mat: np.ndarray, task: str) -> float:
    if task == "classification":
        p = model.predict_proba(X_mat)[0]
        return float(p[np.argmax(p)])
    return float(model.predict(X_mat)[0])


def _Ef(model, X_train: pd.DataFrame, task: str) -> float:
    if task == "classification":
        return float(model.predict_proba(X_train.values).max(axis=1).mean())
    return float(model.predict(X_train.values).mean())


# Both baseline (epsilon=0) and perturbed (epsilon>0) runs receive the SAME:
#   - instance_indices  -> identical test instances
#   - perm_seeds        -> identical permutation orders
#   - noise_seeds       -> identical SEM Gaussian noise draws
# Only the SEM coefficient values differ between runs.
# Therefore: MAD = |phi_epsilon - phi_0| measures TRUE weight sensitivity only.

def _vc(
    S: Set[str],
    x_obs: pd.Series,
    features: List[str],
    sems: List[PerturbableSEM],
    mechs_list: List[Dict[str, dict]],
    model,
    task: str,
    M_per_sem: int,
    noise_seeds: List[int],
) -> float:
    """Estimate v(S) = E[f(X) | do(X_S = x_S)] via MC over interventional samples."""
    total, count = 0.0, 0
    seed_idx = 0
    for sem, mechs in zip(sems, mechs_list):
        for _ in range(M_per_sem):
            noise_rng = np.random.default_rng(noise_seeds[seed_idx % len(noise_seeds)])
            seed_idx += 1
            xs = sem.sample_intervention(S, x_obs, mechs, noise_rng)
            Xm = pd.DataFrame([xs], columns=features).values
            total += _predict_scalar(model, Xm, task)
            count += 1
    return total / count if count > 0 else 0.0


def compute_instance_shap(
    x: pd.Series,
    features: List[str],
    sems: List[PerturbableSEM],
    mechs_list: List[Dict[str, dict]],
    gamma: Dict[str, float],
    model,
    task: str,
    Ef: float,
    n_perms: int,
    M_per_sem: int,
    perm_seeds: List[int],
    noise_seeds: List[int],
) -> np.ndarray:
    """Compute Causal SHAP phi for one instance.
    perm_seeds and noise_seeds are passed in from the caller and are
    identical for baseline and all perturbed runs on the same instance.
    """
    phi = np.zeros(len(features))
    fx  = _predict_scalar(
        model,
        pd.DataFrame([x.to_dict()], columns=features).values,
        task,
    )

    for perm_idx in range(n_perms):
        perm_rng = np.random.default_rng(perm_seeds[perm_idx])
        perm = perm_rng.permutation(features)

        S: Set[str] = set()
        v_prev = _vc(S, x, features, sems, mechs_list, model, task, M_per_sem, noise_seeds)
        for feat in perm:
            S_next = S | {feat}
            v_curr = _vc(S_next, x, features, sems, mechs_list, model, task, M_per_sem, noise_seeds)
            phi[features.index(feat)] += gamma[feat] * (v_curr - v_prev)
            v_prev, S = v_curr, S_next

    phi /= n_perms
    s = float(np.sum(phi))
    if abs(s) > 1e-12:
        phi = phi * ((fx - Ef) / s)
    return phi


def compute_global_shap(
    X_test: pd.DataFrame,
    features: List[str],
    sems: List[PerturbableSEM],
    mechs_list: List[Dict[str, dict]],
    gamma: Dict[str, float],
    model,
    task: str,
    Ef: float,
    instance_indices: np.ndarray,
    n_perms: int,
    M_per_sem: int,
    perm_seeds: List[int],
    noise_seeds: List[int],
    verbose: bool = False,
) -> np.ndarray:
    """Compute Causal SHAP matrix (k x n_features)."""
    phis = []
    for t, i in enumerate(instance_indices):
        if verbose and (t + 1) % 10 == 0:
            print(f"    instance {t+1}/{len(instance_indices)}", flush=True)
        phi = compute_instance_shap(
            X_test.iloc[i], features, sems, mechs_list, gamma,
            model, task, Ef, n_perms, M_per_sem, perm_seeds, noise_seeds,
        )
        phis.append(phi)
    return np.array(phis)


# 5. GAMMA WEIGHTS (IDA total effects)

def compute_gamma(
    dags: List[nx.DiGraph],
    features: List[str],
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    totals: Dict[str, List[float]] = {f: [] for f in features}
    y_vals = y.values

    for dag in dags:
        G = dag.copy()
        G.add_node("Y")

        for u, v in dag.edges():
            parents = [p for p in dag.predecessors(v) if p != u]
            Xr = (np.column_stack([X[parents].values, X[u].values])
                  if parents else X[[u]].values)
            lr = LinearRegression().fit(Xr, X[v].values)
            w  = float(abs(lr.coef_[-1]))
            if w > 1e-12:
                G.add_edge(u, v, weight=w)

        lrY = LinearRegression().fit(X.values, y_vals)
        for feat, w in zip(features, np.abs(lrY.coef_)):
            if w > 1e-12:
                G.add_edge(feat, "Y", weight=float(w))

        for feat in features:
            total = 0.0
            try:
                for path in nx.all_simple_paths(G, feat, "Y", cutoff=10):
                    prod, ok = 1.0, True
                    for a, b in zip(path[:-1], path[1:]):
                        ed = G.get_edge_data(a, b)
                        if not ed or "weight" not in ed:
                            ok = False; break
                        prod *= ed["weight"]
                    if ok:
                        total += prod
            except Exception:
                pass
            totals[feat].append(total)

    effects = {f: float(np.mean(v)) if v else 0.0 for f, v in totals.items()}
    Z = sum(abs(v) for v in effects.values()) + 1e-12
    return {f: abs(v) / Z for f, v in effects.items()}


# 6. causal graph helpers

def sample_dags_from_cpdag(
    cpdag: np.ndarray,
    features: List[str],
    n_dags: int,
    rng: np.random.Generator,
) -> List[nx.DiGraph]:
    n = len(features)
    undirected = [
        (features[i], features[j])
        for i in range(n) for j in range(i + 1, n)
        if cpdag[i, j] == -1 and cpdag[j, i] == -1
    ]

    dags = []
    for _ in range(n_dags):
        G = nx.DiGraph()
        G.add_nodes_from(features)

        # copy directed edges from CPDAG
        for i in range(n):
            for j in range(n):
                if i != j and cpdag[i, j] == -1 and cpdag[j, i] == 1:
                    G.add_edge(features[i], features[j])

        # randomly orient undirected edges
        for u, v in undirected:
            if rng.random() < 0.5:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
            if not nx.is_directed_acyclic_graph(G):
                # flip
                if G.has_edge(u, v):
                    G.remove_edge(u, v); G.add_edge(v, u)
                else:
                    G.remove_edge(v, u); G.add_edge(u, v)
            if not nx.is_directed_acyclic_graph(G):
                # drop the edge entirely
                if G.has_edge(u, v): G.remove_edge(u, v)
                if G.has_edge(v, u): G.remove_edge(v, u)

        if not nx.is_directed_acyclic_graph(G):
            for a, b in list(G.edges()):
                G.remove_edge(a, b)
                if nx.is_directed_acyclic_graph(G):
                    break
                G.add_edge(a, b)

        dags.append(G)
    return dags


def robustness_sweep(
    baseline_phi: np.ndarray,
    X_test: pd.DataFrame,
    features: List[str],
    sems: List[PerturbableSEM],
    gamma: Dict[str, float],
    model,
    task: str,
    Ef: float,
    epsilons: List[float],
    n_trials: int,
    instance_indices: np.ndarray,
    n_perms: int,
    M_per_sem: int,
    perm_seeds: List[int],
    noise_seeds: List[int],
    master_rng: np.random.Generator,
    verbose: bool = True,
) -> List[dict]:
    """For each epsilon: - Repeat n_trials times: 1."""
    phi0_mean = baseline_phi.mean(axis=0)
    rows = []

    for eps in epsilons:
        if verbose:
            print(f"  epsilon={eps:.3f} ...", end="", flush=True)

        trial_mads, trial_rhos, trial_feat_diffs = [], [], []

        for _ in range(n_trials):
            # Draw a fresh perturbation RNG for this trial.
            # This is the ONLY random element that differs from the baseline run.
            pert_rng = np.random.default_rng(int(master_rng.integers(0, 2**32)))
            perturbed_mechs = [sem.perturbed_mechs(eps, pert_rng) for sem in sems]

            phi_eps = compute_global_shap(
                X_test, features, sems, perturbed_mechs, gamma,
                model, task, Ef,
                instance_indices=instance_indices,
                n_perms=n_perms,
                M_per_sem=M_per_sem,
                perm_seeds=perm_seeds,
                noise_seeds=noise_seeds,
                verbose=False,
            )
            phi_eps_mean = phi_eps.mean(axis=0)

            diff = np.abs(phi_eps_mean - phi0_mean)
            trial_mads.append(diff.mean())
            trial_feat_diffs.append(diff)

            rho, _ = spearmanr(np.abs(phi0_mean), np.abs(phi_eps_mean))
            trial_rhos.append(float(rho) if not np.isnan(rho) else 1.0)

        rows.append({
            "epsilon":         eps,
            "mad_mean":        float(np.mean(trial_mads)),
            "mad_std":         float(np.std(trial_mads)),
            "per_feature_mad": np.stack(trial_feat_diffs).mean(axis=0),
            "spearman_mean":   float(np.mean(trial_rhos)),
            "spearman_std":    float(np.std(trial_rhos)),
        })

        if verbose:
            r = rows[-1]
            print(
                f"  MAD={r['mad_mean']:.5f} +/- {r['mad_std']:.5f}"
                f"  rho={r['spearman_mean']:.4f} +/- {r['spearman_std']:.4f}"
            )

    return rows


# 8. monotonicity check

def check_monotonicity(rows: List[dict]) -> None:
    """Checks if mad increases monotonically with epsilon.
    Non-monotonicity means residual MC noise still dominates.
    Only checks rows with epsilon > 0.
    """
    nonzero_rows = [r for r in rows if r["epsilon"] > 0]
    eps_vals = [r["epsilon"]  for r in nonzero_rows]
    mad_vals = [r["mad_mean"] for r in nonzero_rows]

    violations = []
    for i in range(1, len(mad_vals)):
        # Allow 5% tolerance for trial-to-trial variance
        if mad_vals[i] < mad_vals[i - 1] * 0.95:
            violations.append(
                (eps_vals[i - 1], eps_vals[i], mad_vals[i - 1], mad_vals[i])
            )

    if violations:
        print("\n  WARNING: MONOTONICITY VIOLATIONS DETECTED.")
        print("  MAD should increase with epsilon but decreased at:")
        for v in violations:
            print(f"    epsilon={v[0]:.3f} -> {v[1]:.3f}:  MAD {v[2]:.5f} -> {v[3]:.5f}  (DROPPED)")
        print("  This means residual MC noise still dominates.")
        print("  Fix: increase n_perms and M_per_sem further.\n")
    else:
        print("\n  OK: MAD is monotone with epsilon. Results are reliable.\n")


# 9. plots

def plot_mad_curve(
    rows: List[dict],
    features: List[str],
    name: str,
    phi0_mean: np.ndarray,
    noise_floor: Optional[float] = None,
) -> Tuple[plt.Figure, Optional[float]]:
    """Sensitivity curve: mad vs epsilon (left) and noise-corrected mad (right)."""
    eps_vals  = [r["epsilon"]  for r in rows]
    mad_means = [r["mad_mean"] for r in rows]
    mad_stds  = [r["mad_std"]  for r in rows]

    ref       = np.abs(phi0_mean).mean()
    threshold = 0.05 * ref if ref > 1e-10 else None

    # eps_star computed from corrected MAD
    eps_star = None
    if threshold:
        for r in rows:
            if r["epsilon"] == 0.0:
                continue
            corrected = max(0.0, r["mad_mean"] - (noise_floor or 0.0))
            if corrected > threshold:
                eps_star = r["epsilon"]
                break

    has_correction = noise_floor is not None
    fig, axes = plt.subplots(1, 2 if has_correction else 1,
                              figsize=(14 if has_correction else 8, 4.5))
    if not has_correction:
        axes = [axes]

    ax = axes[0]
    ax.plot(eps_vals, mad_means, "o-", color="#185FA5",
            linewidth=2, markersize=6, label="MAD mean")
    lo = np.array(mad_means) - np.array(mad_stds)
    hi = np.array(mad_means) + np.array(mad_stds)
    ax.fill_between(eps_vals, lo, hi, alpha=0.18, color="#185FA5", label="+/- 1 std")
    if noise_floor is not None:
        ax.axhline(noise_floor, color="#888780", linestyle=":",
                   linewidth=1.4, label=f"noise floor (eps=0): {noise_floor:.5f}")
    if threshold:
        ax.axhline(threshold, color="#A32D2D", linestyle="--",
                   linewidth=1.2, label=f"5% of |phi0|: {threshold:.5f}")
    ax.set_xlabel("epsilon", fontsize=12)
    ax.set_ylabel("MAD(phi_eps - phi_0)", fontsize=12)
    ax.set_title(f"Raw MAD vs epsilon — {name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    if has_correction:
        ax2 = axes[1]
        corrected = [max(0.0, r["mad_mean"] - noise_floor) for r in rows]
        ax2.plot(eps_vals, corrected, "s-", color="#0F6E56",
                 linewidth=2, markersize=6, label="MAD - noise floor")
        if threshold:
            ax2.axhline(threshold, color="#A32D2D", linestyle="--",
                        linewidth=1.2, label=f"5% of |phi0|: {threshold:.5f}")
        if eps_star:
            ax2.axvline(eps_star, color="#A32D2D", linestyle=":",
                        linewidth=1.4, label=f"eps* = {eps_star:.2f}")
        ax2.set_xlabel("epsilon", fontsize=12)
        ax2.set_ylabel("Noise-corrected MAD", fontsize=12)
        ax2.set_title(f"Corrected MAD vs epsilon — {name}", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Causal SHAP Robustness — {name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, eps_star


def plot_feature_heatmap(
    rows: List[dict], features: List[str], name: str
) -> plt.Figure:
    """Per-feature |delta phi_i| heatmap. Rows sorted by total sensitivity."""
    eps_vals = [r["epsilon"] for r in rows]
    mat = np.stack([r["per_feature_mad"] for r in rows], axis=1)
    order  = np.argsort(-mat.sum(axis=1))
    mat_s  = mat[order]
    feat_s = [features[i] for i in order]

    fig, ax = plt.subplots(
        figsize=(max(8, len(eps_vals) * 1.2), max(4, len(features) * 0.55))
    )
    im = ax.imshow(mat_s, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(eps_vals)))
    ax.set_xticklabels([f"e={e:.2f}" for e in eps_vals],
                       fontsize=10, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(feat_s)))
    ax.set_yticklabels(feat_s, fontsize=10)
    ax.set_title(f"Per-feature |delta phi_i| heatmap — {name}",
                 fontsize=13, fontweight="bold", pad=10)
    for i in range(mat_s.shape[0]):
        for j in range(mat_s.shape[1]):
            ax.text(
                j, i, f"{mat_s[i, j]:.4f}", ha="center", va="center",
                fontsize=8,
                color="white" if mat_s[i, j] > mat_s.max() * 0.6 else "black",
            )
    plt.colorbar(im, ax=ax, label="|delta phi_i|")
    fig.tight_layout()
    return fig


def plot_rank_stability(rows: List[dict], name: str) -> plt.Figure:
    """Spearman rho of feature rankings vs epsilon."""
    eps_vals = [r["epsilon"]       for r in rows]
    rho_mean = [r["spearman_mean"] for r in rows]
    rho_std  = [r["spearman_std"]  for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eps_vals, rho_mean, "s-", color="#0F6E56",
            linewidth=2, markersize=6, label="Spearman rho mean")
    lo = np.array(rho_mean) - np.array(rho_std)
    hi = np.array(rho_mean) + np.array(rho_std)
    ax.fill_between(eps_vals, lo, hi, alpha=0.18, color="#0F6E56", label="+/- 1 std")
    ax.axhline(0.9, color="#854F0B", linestyle="--",
               linewidth=1.2, label="rho=0.90 (strong agreement)")
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel("epsilon", fontsize=12)
    ax.set_ylabel("Spearman rho (rank stability)", fontsize=12)
    ax.set_title(f"Feature-rank stability vs epsilon — {name}",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_grouped_bar(
    rows: List[dict], features: List[str], name: str, top_k: int = 10
) -> plt.Figure:
    """Grouped bar: top-k most epsilon-sensitive features, bars grouped by epsilon."""
    eps_vals  = [r["epsilon"] for r in rows]
    mat       = np.stack([r["per_feature_mad"] for r in rows], axis=1)
    top_idx   = np.argsort(-mat.max(axis=1))[:top_k]
    top_feats = [features[i] for i in top_idx]
    top_mat   = mat[top_idx]

    n_feats, n_eps = len(top_feats), len(eps_vals)
    x     = np.arange(n_feats)
    width = 0.7 / n_eps
    cmap  = plt.colormaps["Blues"]
    colors = [cmap(0.3 + 0.6 * i / max(n_eps - 1, 1)) for i in range(n_eps)]

    fig, ax = plt.subplots(figsize=(max(10, n_feats * 1.3), 5))
    for j, (eps, color) in enumerate(zip(eps_vals, colors)):
        offsets = x + (j - n_eps / 2.0 + 0.5) * width
        ax.bar(offsets, top_mat[:, j], width=width * 0.9,
               label=f"e={eps:.2f}", color=color, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(top_feats, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("|delta phi_i| (mean across trials)", fontsize=11)
    ax.set_title(f"Top-{n_feats} epsilon-sensitive features — {name}",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, ncol=min(n_eps, 5))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_noise_floor_comparison(all_results: dict) -> plt.Figure:
    """Cross-dataset stacked bar: noise floor vs true sensitivity at each epsilon.
    Only meaningful when epsilon=0 was included in the sweep.
    """
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 4.5), squeeze=False)

    for ax, (name, res) in zip(axes[0], all_results.items()):
        rows        = res["rows"]
        eps_vals    = [r["epsilon"]  for r in rows]
        mad_means   = [r["mad_mean"] for r in rows]
        noise_floor = res.get("noise_floor", 0.0)
        corrected   = [max(0.0, m - noise_floor) for m in mad_means]

        x = np.arange(len(eps_vals))
        ax.bar(x, [noise_floor] * len(eps_vals),
               color="salmon",   alpha=0.8, label=f"noise floor ({noise_floor:.4f})")
        ax.bar(x, corrected, bottom=[noise_floor] * len(eps_vals),
               color="seagreen", alpha=0.8, label="true sensitivity")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e:.2f}" for e in eps_vals],
                           rotation=30, ha="right", fontsize=9)
        ax.set_xlabel("epsilon", fontsize=11)
        ax.set_ylabel("MAD", fontsize=11)
        ax.set_title(f"{name}\nnoise floor vs true sensitivity",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Noise floor decomposition — all datasets",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# 10. main pipeline - one dataset

def run_robustness_analysis_single(
    csv_path: str,
    name: str = "dataset",
    target_hint: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> dict:
    if cfg is None:
        cfg = ROBUSTNESS_CONFIG

    master_rng = np.random.default_rng(cfg["random_seed"])

    print(f"\n{'='*70}")
    print(f"  Robustness analysis: {name}  ({csv_path})")
    print(f"{'='*70}")

    raw = _drop_unnamed(pd.read_csv(csv_path))
    tgt = _detect_target(raw, target_hint)
    y_raw = raw[tgt]
    X_raw = _to_numeric(raw.drop(columns=[tgt]))
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(
        X_raw.median(numeric_only=True)
    )

    y_type = type_of_target(y_raw)
    task   = "classification" if y_type in ["binary", "multiclass"] else "regression"
    y_vec  = (y_raw.astype(int).values if task == "classification"
              else pd.to_numeric(y_raw, errors="coerce").values)

    strat = y_vec if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y_vec,
        test_size=cfg["test_size"],
        random_state=cfg["random_seed"],
        stratify=strat,
    )
    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    Xte = pd.DataFrame(scaler.transform(X_te),     columns=X_te.columns, index=X_te.index)
    features = list(Xtr.columns)

    print(f"Task: {task}  |  {len(features)} features  |  "
          f"{len(Xtr)} train / {len(Xte)} test samples")

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=160, max_depth=6, random_state=cfg["random_seed"])
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=cfg["random_seed"])
    model.fit(Xtr.values, y_tr)
    print("OK: model trained")

    print("Running PC causal discovery ...")
    from causallearn.search.ConstraintBased.PC import pc as pc_alg
    res    = pc_alg(Xtr.values, alpha=cfg["alpha_pc"], show_progress=False)
    cpdag  = res.G.graph
    dags   = sample_dags_from_cpdag(cpdag, features, cfg["n_dags"], master_rng)
    print(f"OK: sampled {len(dags)} DAGs")

    sems = [PerturbableSEM(d, features).fit(Xtr) for d in dags]
    print("OK: SEMs fitted")

    gamma = compute_gamma(dags, features, Xtr, pd.Series(y_tr))
    Ef    = _Ef(model, Xtr, task)
    print("OK: gamma weights computed")
    for f in features:
        print(f"  {f:25s}  gamma={gamma[f]:.6f}")

    # Generated ONCE, reused for EVERY run (baseline + all epsilon values).
    # This ensures that baseline and perturbed SHAP estimates differ ONLY
    # due to the edge-weight perturbation, not due to different:
    #   - test instance selection
    #   - permutation orders
    #   - SEM Gaussian noise draws
    k = min(cfg["k_instances"], len(Xte))
    instance_indices = master_rng.choice(len(Xte), size=k, replace=False)
    perm_seeds  = [int(master_rng.integers(0, 2**32)) for _ in range(cfg["n_perms"])]
    noise_seeds = [int(master_rng.integers(0, 2**32))
                   for _ in range(cfg["n_dags"] * cfg["M_per_sem"] * 10)]

    print(f"\nOK: fixed {k} instances, {cfg['n_perms']} perm seeds, "
          f"{len(noise_seeds)} noise seeds for ALL runs")

    print(f"\nComputing baseline global SHAP "
          f"(k={k}, n_perms={cfg['n_perms']}, M_per_sem={cfg['M_per_sem']}) ...")
    base_mechs = [sem.base_mechs for sem in sems]
    baseline_phi = compute_global_shap(
        Xte, features, sems, base_mechs, gamma, model, task, Ef,
        instance_indices=instance_indices,
        n_perms=cfg["n_perms"],
        M_per_sem=cfg["M_per_sem"],
        perm_seeds=perm_seeds,
        noise_seeds=noise_seeds,
        verbose=True,
    )
    phi0_mean = baseline_phi.mean(axis=0)

    print("Baseline phi_0 (sorted by |phi|):")
    for f, v in sorted(zip(features, phi0_mean), key=lambda x: -abs(x[1])):
        print(f"  {f:25s}  {v:+.6f}")

    print(f"\nRunning robustness sweep over epsilon={cfg['epsilons']} "
          f"({cfg['n_trials']} trials each) ...")
    rows = robustness_sweep(
        baseline_phi=baseline_phi,
        X_test=Xte,
        features=features,
        sems=sems,
        gamma=gamma,
        model=model,
        task=task,
        Ef=Ef,
        epsilons=cfg["epsilons"],
        n_trials=cfg["n_trials"],
        instance_indices=instance_indices,
        n_perms=cfg["n_perms"],
        M_per_sem=cfg["M_per_sem"],
        perm_seeds=perm_seeds,
        noise_seeds=noise_seeds,
        master_rng=master_rng,
        verbose=True,
    )

    noise_floor = None
    eps0_row = next((r for r in rows if r["epsilon"] == 0.0), None)
    if eps0_row:
        noise_floor = eps0_row["mad_mean"]
        ref_mad = np.abs(phi0_mean).mean()
        print(f"\n  MC noise floor (MAD at epsilon=0): {noise_floor:.6f}")
        if ref_mad > 1e-10 and noise_floor > 0.01 * ref_mad:
            print(f"  WARNING: noise floor is {100*noise_floor/ref_mad:.1f}% of |phi_0|.")
            print("           Consider increasing n_perms / M_per_sem.")
        else:
            print("  OK: noise floor is negligible. Results are trustworthy.")

    check_monotonicity(rows)

    summary_records = []
    for r in rows:
        rec = {
            "epsilon":         r["epsilon"],
            "mad_mean":        r["mad_mean"],
            "mad_std":         r["mad_std"],
            "mad_corrected":   max(0.0, r["mad_mean"] - (noise_floor or 0.0)),
            "spearman_mean":   r["spearman_mean"],
            "spearman_std":    r["spearman_std"],
        }
        for i, f in enumerate(features):
            rec[f"mad_{f}"] = r["per_feature_mad"][i]
        summary_records.append(rec)

    summary = pd.DataFrame(summary_records).set_index("epsilon")
    summary.to_csv(f"robustness_scores_{name}.csv")
    print(f"\nSaved: robustness_scores_{name}.csv")
    print(
        summary[["mad_mean", "mad_std", "mad_corrected",
                 "spearman_mean", "spearman_std"]].round(6).to_string()
    )

    ref       = np.abs(phi0_mean).mean()
    threshold = 0.05 * ref if ref > 1e-10 else None
    eps_star  = None
    if threshold:
        for r in rows:
            if r["epsilon"] == 0.0:
                continue
            corrected = max(0.0, r["mad_mean"] - (noise_floor or 0.0))
            if corrected > threshold:
                eps_star = r["epsilon"]
                break

    if eps_star:
        print(f"\n  Breakdown eps* (corrected MAD > 5% of |phi_0|): eps* = {eps_star:.2f}")
    else:
        print("\n  Attributions stable across all tested epsilon values (after noise correction).")

    fig_mad, _ = plot_mad_curve(rows, features, name, phi0_mean, noise_floor)
    fig_mad.savefig(f"robustness_mad_{name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: robustness_mad_{name}.png")

    fig_hm = plot_feature_heatmap(rows, features, name)
    fig_hm.savefig(f"robustness_heatmap_{name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: robustness_heatmap_{name}.png")

    fig_rk = plot_rank_stability(rows, name)
    fig_rk.savefig(f"robustness_rank_{name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: robustness_rank_{name}.png")

    fig_bar = plot_grouped_bar(rows, features, name)
    fig_bar.savefig(f"robustness_bar_{name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: robustness_bar_{name}.png")

    plt.close("all")

    return {
        "summary":      summary,
        "baseline_phi": baseline_phi,
        "phi0_mean":    phi0_mean,
        "features":     features,
        "noise_floor":  noise_floor,
        "eps_star":     eps_star,
        "rows":         rows,
    }


# 11. main entry point - all datasets

def run_robustness_analysis(
    csv_configs: List[dict],
    cfg: Optional[dict] = None,
) -> dict:
    """Run robustness analysis for one or more datasets."""
    if cfg is None:
        cfg = ROBUSTNESS_CONFIG

    all_results = {}
    for c in csv_configs:
        result = run_robustness_analysis_single(
            csv_path=c["path"],
            name=c.get("name", c["path"]),
            target_hint=c.get("target_hint"),
            cfg=cfg,
        )
        all_results[c.get("name", c["path"])] = result

    # Cross-dataset noise-floor comparison (only if epsilon=0 was in the sweep)
    if 0.0 in cfg["epsilons"]:
        fig_nf = plot_noise_floor_comparison(all_results)
        fig_nf.savefig("robustness_noise_floor_comparison.png",
                       dpi=150, bbox_inches="tight")
        print("\nSaved: robustness_noise_floor_comparison.png")

    print("\n" + "="*70)
    print("  ROBUSTNESS ANALYSIS COMPLETE")
    print("="*70)
    for name, res in all_results.items():
        nf  = res.get("noise_floor")
        es  = res.get("eps_star")
        phi = res["phi0_mean"]
        dom = res["features"][np.argmax(np.abs(phi))]
        print(f"\n  {name}:")
        print(f"    dominant feature : {dom}  phi_0={phi[np.argmax(np.abs(phi))]:+.4f}")
        if nf is not None:
            print(f"    MC noise floor   : {nf:.6f}")
        if es:
            print(f"    breakdown eps*   : {es:.2f}  (corrected for noise floor)")
        else:
            print(f"    breakdown eps*   : stable across all epsilon values")

    plt.close("all")
    return all_results


# 12. script entry point

if __name__ == "__main__":
    run_robustness_analysis([
        {
            "path": "../datasets/cv_risk.csv",
            "name": "cv_risk",
        },
        {
            "path": "../datasets/lung_cancer_risk.csv",
            "name": "lung_cancer_risk",
        },
    ])
