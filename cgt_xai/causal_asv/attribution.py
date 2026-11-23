"""PC causal discovery followed by asymmetric Shapley attribution."""

import numpy as np
import pandas as pd
from typing import Callable, Optional, List, Dict
import networkx as nx
from .causal_discovery import PCAlgorithm
from .asymmetric_shapley import AsymmetricShapleyValues


class CausalASVAnalyzer:
    """Unified causal-aware feature attribution analyzer."""

    def __init__(self, model: Callable,
                 significance_level: float = 0.05,
                 background_data: Optional[pd.DataFrame] = None):
        self.model = model
        self.significance_level = significance_level
        self.background_data = background_data
        self.causal_graph = None
        self.causal_order = None
        self.pc_algorithm = PCAlgorithm(significance_level=significance_level)
        self.asv = None

    def discover_causal_structure(self, data: pd.DataFrame,
                                   verbose: bool = True) -> nx.DiGraph:
        """Discover causal structure from observational data."""
        self.causal_graph = self.pc_algorithm.fit(data, verbose=verbose)
        self.causal_order = self.pc_algorithm.get_causal_order()

        # Initialize ASV with discovered graph
        self.asv = AsymmetricShapleyValues(
            self.model,
            self.causal_graph,
            self.background_data
        )

        return self.causal_graph

    def shapley_values(self, X: pd.DataFrame,
                      max_samples: Optional[int] = None,
                      feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute Asymmetric Shapley Values for instances."""
        if self.asv is None:
            raise ValueError(
                "Must call discover_causal_structure() first or provide causal_graph"
            )

        attributions = self.asv.shapley_values(
            X,
            max_samples=max_samples,
            feature_names=feature_names
        )

        # Convert to DataFrame
        if feature_names is None:
            feature_names = list(X.columns)

        return pd.DataFrame(
            attributions,
            columns=feature_names,
            index=X.index
        )

    def explain(self, X: pd.DataFrame,
                discover_structure: bool = False,
                max_samples: Optional[int] = None) -> Dict:
        """Complete pipeline: discover causal structure and compute attributions."""
        if discover_structure or self.causal_graph is None:
            self.discover_causal_structure(X)

        attributions = self.shapley_values(X, max_samples=max_samples)

        return {
            'attributions': attributions,
            'causal_graph': self.causal_graph,
            'causal_order': self.causal_order
        }

    def set_causal_graph(self, causal_graph: nx.DiGraph):
        """Set causal graph directly (if known a priori).
        
                Parameters
                ----------
                causal_graph : nx.DiGraph
                    Known causal structure
        """
        self.causal_graph = causal_graph
        self.causal_order = list(nx.topological_sort(causal_graph))

        # Reinitialize ASV
        self.asv = AsymmetricShapleyValues(
            self.model,
            self.causal_graph,
            self.background_data
        )

    def get_causal_order(self) -> List[str]:
        """Get causal ordering of features."""
        if self.causal_order is None:
            raise ValueError("No causal structure discovered yet")
        return self.causal_order

    def get_parents(self, feature: str) -> List[str]:
        """Get causal parents of a feature."""
        if self.causal_graph is None:
            raise ValueError("No causal structure discovered yet")
        return list(self.causal_graph.predecessors(feature))

    def get_children(self, feature: str) -> List[str]:
        """Get causal children of a feature."""
        if self.causal_graph is None:
            raise ValueError("No causal structure discovered yet")
        return list(self.causal_graph.successors(feature))

    def visualize_graph(self, **kwargs):
        """Visualize causal graph.
        
                See visualization.py for details.
        """
        try:
            from .visualization import visualize_causal_graph
        except ImportError:
            from .visualization import visualize_causal_graph

        if self.causal_graph is None:
            raise ValueError("No causal structure discovered yet")

        return visualize_causal_graph(self.causal_graph, **kwargs)

    def aggregate_causal_attributions(self, attributions: pd.DataFrame,
                                      feature_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Aggregate attributions by feature groups."""
        try:
            from .utils import aggregate_causal_attributions
        except ImportError:
            from .utils import aggregate_causal_attributions
        return aggregate_causal_attributions(attributions, feature_groups)


def analyze_causal_attribution(model: Callable,
                              X: pd.DataFrame,
                              discovery_data: Optional[pd.DataFrame] = None,
                              significance_level: float = 0.05,
                              max_samples: Optional[int] = None,
                              verbose: bool = True) -> Dict:
    """Convenience function for complete causal attribution analysis."""
    analyzer = CausalASVAnalyzer(
        model,
        significance_level=significance_level
    )

    # Discover causal structure
    if discovery_data is None:
        discovery_data = X

    analyzer.discover_causal_structure(discovery_data, verbose=verbose)

    # Compute attributions
    results = analyzer.explain(X, max_samples=max_samples)

    return results
