"""Causal SHAP.

Runs the PC algorithm (causallearn) to get a CPDAG, samples DAGs consistent with it, fits linear
mechanisms and estimates Shapley values under interventional value functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Optional: Kernel SHAP baseline
try:
    import shap  # type: ignore
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False


def _ensure_2d(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def _prob_to_link(p: float, link: str = 'logit', eps: float = 1e-6) -> float:
    """Map probability to link space.
    - 'prob': identity
    - 'logit': log-odds with clipping to avoid +/-inf
    """
    if link == 'prob':
        return float(p)
    if link == 'logit':
        p = float(np.clip(p, eps, 1.0 - eps))
        return float(np.log(p) - np.log(1.0 - p))
    raise ValueError(f"Unknown link '{link}'. Use 'prob' or 'logit'.")


@dataclass
class CausalStructure:
    """Holds the cpdag (as adjacency matrix) and sampled DAGs."""
    feature_names: List[str]
    cpdag: np.ndarray  # adjacency matrix with causallearn coding (-1, 0, 1)
    dags: List[nx.DiGraph]

def discover_structure(X: pd.DataFrame, alpha: float = 0.1, seed: int = 42) -> CausalStructure:
    """Run PC algorithm (causallearn) to obtain cpdag, then create a CausalStructure stub.
    You can later call `sample_dags` to orient undirected edges.
    """
    from causallearn.search.ConstraintBased.PC import pc  # lazy import
    feature_names = list(X.columns)
    res = pc(X.values, alpha=alpha, show_progress=False)
    cpdag = res.G.graph
    return CausalStructure(feature_names=feature_names, cpdag=cpdag, dags=[])

def sample_dags(structure: CausalStructure, n_samples: int = 5, seed: int = 42) -> List[nx.DiGraph]:
    """Sample multiple DAG orientations consistent with the cpdag. For undirected
    edges, randomly choose a direction and ensure acyclicity by flipping if needed.
    """
    rng = np.random.default_rng(seed)
    names = structure.feature_names
    cpdag = structure.cpdag
    dags: List[nx.DiGraph] = []

    # Identify undirected edge pairs
    undirected_pairs: List[Tuple[str, str]] = []
    directed_pairs: List[Tuple[str, str]] = []
    n = len(names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # directed i -> j (causallearn code: cpdag[i,j]==-1, cpdag[j,i]==1)
            if cpdag[i, j] == -1 and cpdag[j, i] == 1:
                directed_pairs.append((names[i], names[j]))
            # undirected i-j
            if i < j and (cpdag[i, j] == -1 and cpdag[j, i] == -1):
                undirected_pairs.append((names[i], names[j]))

    if not undirected_pairs:
        # already fully directed
        G = nx.DiGraph()
        G.add_nodes_from(names)
        G.add_edges_from(directed_pairs)
        structure.dags = [G]
        return structure.dags

    for _ in range(n_samples):
        G = nx.DiGraph()
        G.add_nodes_from(names)
        # add directed
        for u, v in directed_pairs:
            G.add_edge(u, v)
        # orient undirected
        for u, v in undirected_pairs:
            if rng.random() < 0.5:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
            # ensure DAG by flipping if cycle created
            if not nx.is_directed_acyclic_graph(G):
                if G.has_edge(u, v):
                    G.remove_edge(u, v); G.add_edge(v, u)
                else:
                    G.remove_edge(v, u); G.add_edge(u, v)
        dags.append(G)

    structure.dags = dags
    return dags


class LinearMechanisms:
    """Fit linear mechanisms X_j = a_j + beta_j^T parents + ε_j for DAG sampling under do() interventions."""
    def __init__(self, dag: nx.DiGraph, feature_names: Sequence[str]):
        self.dag = dag
        self.feature_names = list(feature_names)
        self.mechanisms: Dict[str, Dict[str, Union[List[str], float, np.ndarray]]] = {}
        self.topo_order: List[str] = []

    def fit(self, X: pd.DataFrame) -> 'LinearMechanisms':
        # topological order (with fallback)
        try:
            self.topo_order = list(nx.topological_sort(self.dag))
        except Exception:
            self.topo_order = self._safe_topo()

        for var in self.feature_names:
            parents = list(self.dag.predecessors(var))
            if not parents:
                self.mechanisms[var] = {
                    'parents': [],
                    'intercept': float(X[var].mean()),
                    'coef': None,
                    'sigma': float(X[var].std() + 1e-12),
                }
            else:
                lr = LinearRegression().fit(_ensure_2d(X[parents]), X[var].values)
                resid = X[var].values - lr.predict(_ensure_2d(X[parents]))
                self.mechanisms[var] = {
                    'parents': parents,
                    'intercept': float(lr.intercept_),
                    'coef': lr.coef_.astype(float),
                    'sigma': float(np.std(resid) + 1e-12),
                }
        return self

    def _safe_topo(self) -> List[str]:
        ordered: List[str] = []
        remaining = set(self.feature_names)
        while remaining:
            ready = [n for n in remaining if all(p in ordered or p not in remaining for p in self.dag.predecessors(n))]
            if not ready:
                ready = [min(remaining, key=lambda n: len([p for p in self.dag.predecessors(n) if p in remaining]))]
            for n in ready:
                ordered.append(n); remaining.remove(n)
        return ordered

    def sample_intervention(self, S: Set[str], x_obs: pd.Series, rng: np.random.Generator) -> Dict[str, float]:
        """Sample a full feature vector from do(X_S = x_S)."""
        x_new: Dict[str, float] = {}
        for var in self.topo_order:
            if var in S:
                x_new[var] = float(x_obs[var])
            else:
                mech = self.mechanisms[var]
                parents: List[str] = mech['parents']  # type: ignore
                if not parents:
                    mu, sigma = mech['intercept'], mech['sigma']  # type: ignore
                    x_new[var] = float(mu if sigma < 1e-12 else rng.normal(mu, sigma))  # type: ignore
                else:
                    betas: np.ndarray = mech['coef']  # type: ignore
                    mu = mech['intercept'] + np.array([x_new.get(p, float(x_obs[p])) for p in parents]) @ betas  # type: ignore
                    sigma = mech['sigma']  # type: ignore
                    x_new[var] = float(mu if sigma < 1e-12 else rng.normal(mu, sigma))  # type: ignore
        return x_new


class CausalEffects:
    """Estimate total causal effects via path products in augmented graphs (features + Y)
    and compute gamma weights as normalized absolute effects.
    """
    def __init__(self, dags: List[nx.DiGraph], feature_names: Sequence[str]):
        self.dags = dags
        self.feature_names = list(feature_names)
        self.effects: Dict[str, float] = {}

    def compute_effects(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        aug_graphs: List[nx.DiGraph] = []
        # Build augmented graphs with Y node and absolute weights
        for dag in self.dags:
            G = dag.copy(); G.add_node('Y')
            # edges from feature -> Y via conditional regression weight
            for feat in self.feature_names:
                others = [f for f in self.feature_names if f != feat]
                X_cond = np.column_stack([_ensure_2d(X[others]), _ensure_2d(X[[feat]])]) if others else _ensure_2d(X[[feat]])
                lr = LinearRegression().fit(X_cond, y.values)
                w = abs(float(lr.coef_[-1]))
                if w > 1e-12:
                    G.add_edge(feat, 'Y', weight=w)
            # feature->feature weights (absolute)
            for u, v in dag.edges():
                parents = [p for p in dag.predecessors(v) if p != u]
                X_reg = np.column_stack([_ensure_2d(X[parents]), _ensure_2d(X[[u]])]) if parents else _ensure_2d(X[[u]])
                lr = LinearRegression().fit(X_reg, X[v].values)
                w = abs(float(lr.coef_[-1]))
                G[u][v]['weight'] = w
            aug_graphs.append(G)

        totals: Dict[str, List[float]] = {f: [] for f in self.feature_names}
        for G in aug_graphs:
            effs: Dict[str, float] = {}
            for feat in self.feature_names:
                s = 0.0
                try:
                    for path in nx.all_simple_paths(G, feat, 'Y', cutoff=10):
                        prod = 1.0
                        for i in range(len(path) - 1):
                            data = G.get_edge_data(path[i], path[i+1])
                            if data and 'weight' in data:
                                prod *= float(data['weight'])
                        s += prod
                except Exception:
                    pass
                effs[feat] = float(s)
            for k, v in effs.items():
                totals[k].append(v)
        self.effects = {k: float(np.mean(vs)) for k, vs in totals.items()}
        return self.effects

    def gamma_weights(self) -> Dict[str, float]:
        abs_eff = {k: abs(v) for k, v in self.effects.items()}
        tot = sum(abs_eff.values()) + 1e-12
        return {k: v / tot for k, v in abs_eff.items()}


class CausalSHAP:
    """Causal SHAP that supports probability or log-odds value functions."""
    def __init__(self,
                 model,
                 dags: List[nx.DiGraph],
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 link: str = 'logit',
                 m_resamples: int = 64,
                 seed: int = 42):
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model must implement predict_proba for Causal SHAP.")
        self.model = model
        self.dags = dags
        self.X_train = X_train
        self.y_train = y_train
        self.link = link
        self.m_resamples = int(m_resamples)
        self.feature_names = list(X_train.columns)
        self.rng = np.random.default_rng(seed)

        # mechanisms
        self.mechanisms = [LinearMechanisms(d, self.feature_names).fit(X_train) for d in self.dags]
        # effects & gamma
        self.effects = CausalEffects(dags, self.feature_names)
        self.effects.compute_effects(X_train, y_train)
        self.gamma = self.effects.gamma_weights()

    def _value(self, S: Set[str], x: pd.Series, pred_class: int) -> float:
        total = 0.0
        m_per_dag = max(1, self.m_resamples // max(1, len(self.mechanisms)))
        for mech in self.mechanisms:
            for _ in range(m_per_dag):
                x_s = mech.sample_intervention(S, x, self.rng)
                X_mat = pd.DataFrame([x_s], columns=self.feature_names).values
                probs = self.model.predict_proba(X_mat)[0]
                p_k = float(probs[pred_class])
                total += _prob_to_link(p_k, self.link, eps=1e-6)
        denom = m_per_dag * max(1, len(self.mechanisms))
        return total / denom

    def explain_instance(self, x: pd.Series, max_perm: int = 200) -> pd.Series:
        """Return a Series of Causal SHAP values for one instance in chosen link space."""
        probs = self.model.predict_proba(pd.DataFrame([x], columns=self.feature_names).values)[0]
        k = int(np.argmax(probs))
        phi = np.zeros(len(self.feature_names), dtype=float)
        for _ in range(int(max_perm)):
            perm = self.rng.permutation(self.feature_names)
            S: Set[str] = set()
            v_prev = self._value(S, x, k)
            for feat in perm:
                S_next = S | {feat}
                v_curr = self._value(S_next, x, k)
                delta = v_curr - v_prev
                phi[self.feature_names.index(feat)] += self.gamma[feat] * delta
                v_prev = v_curr; S = S_next
        phi /= max(1, int(max_perm))
        # local accuracy in link space
        f_x = _prob_to_link(float(probs[k]), self.link, eps=1e-6)
        v_empty = self._value(set(), x, k)
        s = float(np.sum(phi))
        if abs(s) > 1e-12:
            phi = phi * ((f_x - v_empty) / s)
        return pd.Series(phi, index=self.feature_names, name='Causal_SHAP')

    def explain_global(self, X_test: pd.DataFrame, n_samples: int = 30, max_perm: int = 200) -> pd.Series:
        """Average Causal SHAP over a subset of test instances to get a global explanation."""
        n = min(len(X_test), int(n_samples))
        idxs = self.rng.choice(len(X_test), size=n, replace=False)
        exps = [self.explain_instance(X_test.iloc[i], max_perm=max_perm) for i in idxs]
        return pd.concat(exps, axis=1).mean(axis=1).rename('Causal_SHAP')


def plot_causal_graph(G: nx.DiGraph, path: Optional[str] = None, seed: int = 42, title: str = "Discovered Causal Graph"):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=seed, k=1.0, iterations=200)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1200, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15,
                           arrowstyle='->', width=2, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.axis('off'); plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches='tight')
    return path

def plot_comparison(df_comp: pd.DataFrame, title: str, path: Optional[str] = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Bar: first column
    axes[0, 0].bar(np.arange(df_comp.shape[0]), df_comp.iloc[:, 0], width=0.6)
    axes[0, 0].set_title(f'{title}\n(Bar: first method)')
    axes[0, 0].set_ylabel('Attribution (link space)')
    axes[0, 0].set_xlabel('Feature')
    axes[0, 0].set_xticks(np.arange(df_comp.shape[0]))
    axes[0, 0].set_xticklabels(df_comp.index, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    # Heatmap
    im = axes[0, 1].imshow(df_comp.T.values, aspect='auto')
    axes[0, 1].set_title('Attribution Heatmap')
    axes[0, 1].set_yticks(np.arange(df_comp.T.shape[0]))
    axes[0, 1].set_yticklabels(df_comp.columns)
    axes[0, 1].set_xticks(np.arange(df_comp.shape[0]))
    axes[0, 1].set_xticklabels(df_comp.index, rotation=45, ha='right')
    cbar = plt.colorbar(im, ax=axes[0, 1]); cbar.set_label('Score')
    # Normalized comparison
    df_norm = df_comp.div(df_comp.abs().max(axis=0), axis=1)
    for i, col in enumerate(df_norm.columns):
        axes[1, 0].bar(np.arange(df_norm.shape[0]) + (i-0.5)*(0.8/len(df_norm.columns)), df_norm[col], width=0.8/len(df_norm.columns), label=col)
    axes[1, 0].set_title('Normalized (by method max |value|)'); axes[1, 0].legend(fontsize=9)
    axes[1, 0].set_xticks(np.arange(df_norm.shape[0])); axes[1, 0].set_xticklabels(df_norm.index, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    # Rank heatmap
    ranks = df_comp.rank(ascending=False)
    im2 = axes[1, 1].imshow(ranks.T.values, aspect='auto')
    axes[1, 1].set_title('Ranking'); cbar2 = plt.colorbar(im2, ax=axes[1, 1]); cbar2.set_label('Rank')
    axes[1, 1].set_yticks(np.arange(ranks.T.shape[0])); axes[1, 1].set_yticklabels(df_comp.columns)
    axes[1, 1].set_xticks(np.arange(df_comp.shape[0])); axes[1, 1].set_xticklabels(df_comp.index, rotation=45, ha='right')
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def model_feature_importance(model, feature_names: Sequence[str]) -> Optional[pd.Series]:
    """Return RandomForest-like native importances if present (normalized to sum to 1)."""
    if hasattr(model, 'feature_importances_'):
        return pd.Series(getattr(model, 'feature_importances_'), index=list(feature_names), name='Model_Importance')
    return None

def kernel_shap_global(model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Optional[pd.Series]:
    """Return Kernel SHAP global importance (mean |value| across samples and classes) if shap is installed."""
    if not HAVE_SHAP:
        return None
    try:
        bg = shap.kmeans(X_train.values, min(100, len(X_train)))
        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x), bg)
        sv = explainer.shap_values(X_test.values, nsamples=100)
        if isinstance(sv, list):
            arr = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
        else:
            arr = np.abs(sv).mean(axis=0)
        return pd.Series(arr, index=X_test.columns, name='Kernel_SHAP')
    except Exception:
        return None


def _choose_link_for_alignment(include_kernel_shap: bool, kernel_shap_ok: bool, default_link: str = 'prob') -> str:
    """Heuristic: if Kernel SHAP is present (probability space), choose 'prob' for alignment;
    otherwise use default.
    """
    if include_kernel_shap and kernel_shap_ok:
        return 'prob'
    return default_link

def run_causal_shap_global_aligned(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    *,
    link: Optional[str] = None,       # Force link if provided
    align: str = 'auto',               # 'auto' | 'prob' | 'logit'
    alpha_pc: float = 0.1,
    n_dags: int = 5,
    max_perm: int = 200,
    m_resamples: int = 64,
    n_global_samples: int = 30,
    seed: int = 42,
    save_prefix: str = 'experiment',
    include_kernel_shap: bool = False,
    default_link_if_auto: str = 'prob'
) -> Tuple[pd.DataFrame, str, str, str]:
    """Compute global Causal SHAP (prob or logit), align to other methods if requested, save comparison artifacts, and return (df, graph_png, comp_png, used_link)."""
    # Optionally precompute Kernel SHAP to decide alignment
    kernel_series = None
    kernel_ok = False
    if include_kernel_shap:
        kernel_series = kernel_shap_global(model, X_train, X_test)
        kernel_ok = kernel_series is not None

    # Decide link
    if link is not None:
        used_link = link
    else:
        if align == 'auto':
            used_link = _choose_link_for_alignment(include_kernel_shap, kernel_ok, default_link_if_auto)
        elif align in ('prob', 'logit'):
            used_link = align
        else:
            raise ValueError("align must be 'auto', 'prob', or 'logit'")

    # 1) Causal discovery + DAGs
    structure = discover_structure(X_train, alpha=alpha_pc, seed=seed)
    dags = sample_dags(structure, n_samples=n_dags, seed=seed)

    # 2) Causal SHAP with chosen link
    cs = CausalSHAP(model, dags, X_train, y_train, link=used_link, m_resamples=m_resamples, seed=seed)
    causal_global = cs.explain_global(X_test, n_samples=n_global_samples, max_perm=max_perm)

    # 3) Baselines
    attributions: Dict[str, pd.Series] = {'Causal_SHAP': causal_global}
    fi = model_feature_importance(model, X_train.columns)
    if fi is not None:
        attributions['Model_Importance'] = fi
    if include_kernel_shap and kernel_ok and kernel_series is not None:
        attributions['Kernel_SHAP'] = kernel_series

    df = pd.DataFrame(attributions)

    # 4) Save outputs
    csv_path = f'attribution_scores_{save_prefix}.csv'
    df.to_csv(csv_path, index=True)
    graph_png = f'causal_graph_{save_prefix}.png'
    comp_png  = f'attribution_comparison_{save_prefix}.png'
    # draw base digraph for visualization
    base_G = nx.DiGraph(); base_G.add_nodes_from(structure.feature_names)
    # Add only directed edges for clarity
    n = len(structure.feature_names)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if structure.cpdag[i, j] == -1 and structure.cpdag[j, i] == 1:
                base_G.add_edge(structure.feature_names[i], structure.feature_names[j])
    plot_causal_graph(base_G, path=graph_png, seed=seed, title=f'Discovered Causal Graph (PC, alpha={alpha_pc})')
    plot_comparison(df, title=f'Feature Attribution Comparison ({save_prefix}, link={used_link})', path=comp_png)

    return df, graph_png, comp_png, used_link


if __name__ == "__main__":
    import argparse
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score

    parser = argparse.ArgumentParser(description="Run Causal SHAP on an NPZ dataset.")
    parser.add_argument("--npz", type=str, required=True, help="Path to NPZ with X_train, X_test, y_train, y_test, feature_names")
    parser.add_argument("--link", type=str, default="logit", choices=["prob", "logit"])
    parser.add_argument("--align", type=str, default="auto", choices=["auto", "prob", "logit"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_dags", type=int, default=5)
    parser.add_argument("--max_perm", type=int, default=200)
    parser.add_argument("--m_resamples", type=int, default=64)
    parser.add_argument("--n_global", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="experiment")
    parser.add_argument("--kernel", action="store_true", help="Include Kernel SHAP baseline if shap is installed.")
    args = parser.parse_args()

    print("\n=== Causal SHAP CLI Demo ===")
    data = np.load(args.npz, allow_pickle=True)
    req = {'X_train', 'X_test', 'y_train', 'y_test', 'feature_names'}
    if not req.issubset(set(data.keys())):
        raise RuntimeError(f"NPZ must contain: {req}")

    X_train = pd.DataFrame(data['X_train'], columns=list(data['feature_names']))
    X_test  = pd.DataFrame(data['X_test'],  columns=list(data['feature_names']))
    y_train = pd.Series(data['y_train'].astype(int))
    y_test  = pd.Series(data['y_test'].astype(int))

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    model = RandomForestClassifier(n_estimators=160, max_depth=6, random_state=args.seed)
    model.fit(X_train_s.values, y_train.values)
    y_pred = model.predict(X_test_s.values)
    print(f"Accuracy={accuracy_score(y_test, y_pred):.4f}, MacroF1={f1_score(y_test, y_pred, average='macro'):.4f}")

    df, gpng, cpng, used = run_causal_shap_global_aligned(
        model, X_train_s, X_test_s, y_train,
        link=args.link if args.align != 'auto' else None,
        align=args.align,
        alpha_pc=args.alpha,
        n_dags=args.n_dags,
        max_perm=args.max_perm,
        m_resamples=args.m_resamples,
        n_global_samples=args.n_global,
        seed=args.seed,
        save_prefix=args.prefix,
        include_kernel_shap=args.kernel,
        default_link_if_auto=args.link
    )
    print("\nUsed link:", used)
    print(df.round(6))
    print(f"\nSaved: {gpng}, {cpng}, attribution_scores_{args.prefix}.csv")
