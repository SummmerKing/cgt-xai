"""Asymmetric Shapley values that respect a causal ordering of the features."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Optional
from itertools import combinations, chain
from tqdm import tqdm
import networkx as nx


class AsymmetricShapleyValues:
    """Compute Asymmetric Shapley Values that respect causal ordering."""

    def __init__(self, model: Callable, causal_graph: nx.DiGraph,
                 background_data: Optional[pd.DataFrame] = None):
        self.model = model
        self.causal_graph = causal_graph
        self.background_data = background_data

        # Extract causal ordering
        self.causal_order = self._extract_causal_order()

    def _extract_causal_order(self) -> List[str]:
        """Extract causal ordering from the graph.
        
                Returns
                -------
                List[str]
                    Variables ordered by causal precedence
        """
        try:
            return list(nx.topological_sort(self.causal_graph))
        except nx.NetworkXError:
            # Graph has cycles, use approximate ordering
            in_degrees = dict(self.causal_graph.in_degree())
            return sorted(self.causal_graph.nodes(), key=lambda x: in_degrees.get(x, 0))

    def shapley_values(self, X: pd.DataFrame,
                      max_samples: Optional[int] = None,
                      feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Compute Asymmetric Shapley Values."""
        X = X.copy()
        if feature_names is None:
            feature_names = list(X.columns)

        n_samples, n_features = X.shape

        # Initialize result matrix
        attributions = np.zeros((n_samples, n_features))

        if self.background_data is None:
            # Use mean as background (repeat single value for sampling)
            background = {col: np.array([X[col].mean()]) for col in X.columns}
        else:
            background = {col: self.background_data[col].values for col in X.columns}

        # Compute attributions for each instance
        for i in tqdm(range(n_samples), desc="Computing ASVs"):
            x_dict = X.iloc[i].to_dict()

            for j, feature in enumerate(feature_names):
                asv_value = self._compute_asv_single(
                    x_dict, feature, background,
                    max_samples=max_samples
                )
                attributions[i, j] = asv_value

        return attributions

    def _compute_asv_single(self, x: dict, feature: str,
                            background: dict,
                            max_samples: Optional[int] = None) -> float:
        """Compute asv for a single feature and instance."""
        if max_samples is None:
            max_samples = min(2 ** len(self.causal_order), 1000)

        # Get features that cause 'feature' (parents in causal graph)
        parents = list(self.causal_graph.predecessors(feature))

        # Get position of feature in causal order
        try:
            feature_idx = self.causal_order.index(feature)
        except ValueError:
            feature_idx = len(self.causal_order)

        # Monte Carlo sampling
        asv_value = 0.0
        n_samples = 0

        for _ in range(max_samples):
            # Sample a random subset that respects causal ordering
            # Only include features that come before 'feature' in causal order
            prev_features = [f for f in self.causal_order[:feature_idx]
                            if f != feature]

            if len(prev_features) == 0:
                subset = set()
            else:
                # Random subset of previous features (respecting causal order)
                k = np.random.randint(0, len(prev_features) + 1)
                subset = set(np.random.choice(
                    prev_features, size=min(k, len(prev_features)),
                    replace=False
                ))

            # Compute marginal contribution
            contribution = self._marginal_contribution(
                x, feature, subset, background
            )

            asv_value += contribution
            n_samples += 1

        return asv_value / n_samples if n_samples > 0 else 0.0

    def _marginal_contribution(self, x: dict, feature: str,
                               subset: set, background: dict) -> float:
        """Compute marginal contribution of adding feature to subset.
        
                v(S ∪ {i}) - v(S) where computation respects causal structure.
        """
        # Create reference instances as DataFrames
        x_with_feature = x.copy()
        x_without_feature = x.copy()

        # Set feature values for coalition
        for var in self.causal_order:
            if var not in x:
                continue

            if var == feature:
                # With feature: keep actual value
                # Without feature: set to background value
                x_without_feature[var] = self._sample_background(var, background)
            elif var in subset:
                # In coalition: keep actual value
                x_with_feature[var] = x[var]
                x_without_feature[var] = x[var]
            else:
                # Not in coalition: sample from background
                x_with_feature[var] = self._sample_background(var, background)
                x_without_feature[var] = self._sample_background(var, background)

        # Convert to DataFrame and predict
        X_with = pd.DataFrame([x_with_feature])
        X_without = pd.DataFrame([x_without_feature])

        with_val = self._evaluate_model(X_with)
        without_val = self._evaluate_model(X_without)

        return with_val - without_val

    def _sample_background(self, var: str, background: dict) -> float:
        """Sample a value for variable from background distribution."""
        if var not in background:
            return 0.0
        values = background[var]
        idx = np.random.randint(len(values))
        return float(values[idx])

    def _evaluate_model(self, X: np.ndarray) -> float:
        """Evaluate model on given instances."""
        if isinstance(X, pd.DataFrame):
            pred = self.model(X)
        else:
            # Convert to DataFrame if needed
            if hasattr(self.model, 'feature_names_in_'):
                X_df = pd.DataFrame(X, columns=self.model.feature_names_in_)
            else:
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            pred = self.model(X_df)

        # Return average prediction
        return float(np.mean(pred))


def compute_asymmetric_shapley(model: Callable,
                               X: pd.DataFrame,
                               causal_graph: nx.DiGraph,
                               background_data: Optional[pd.DataFrame] = None,
                               max_samples: Optional[int] = None) -> np.ndarray:
    """Convenience function to compute ASVs."""
    asv = AsymmetricShapleyValues(model, causal_graph, background_data)
    return asv.shapley_values(X, max_samples=max_samples)
