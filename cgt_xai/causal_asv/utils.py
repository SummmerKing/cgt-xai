"""Dataset helpers for the agricultural datasets."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import networkx as nx


# Common agriculture feature groupings
AGRICULTURE_GROUPS = {
    'weather': ['temperature', 'precipitation', 'humidity', 'wind_speed',
                'solar_radiation', 'drought_index'],
    'soil': ['soil_ph', 'soil_moisture', 'organic_matter', 'nitrogen',
             'phosphorus', 'potassium', 'soil_type'],
    'management': ['irrigation', 'fertilizer', 'pesticide', 'tillage',
                   'planting_date', 'harvest_date'],
    'topography': ['elevation', 'slope', 'aspect'],
    'temporal': ['season', 'year', 'month', 'growing_season_length']
}


def get_agriculture_context() -> Dict[str, List[str]]:
    """Get default agriculture feature groupings.
    
        Returns
        -------
        dict
            Feature groups by category
    """
    return AGRICULTURE_GROUPS.copy()


def load_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data from CSV file."""
    import os

    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]
        return X, y
    else:
        # Return simulated data for testing
        print(f"File {filepath} not found. Using simulated data for testing.")
        X, y, _ = simulate_agriculture_data(n_samples=100, n_features=10)
        return X, y


def create_agriculture_causal_prior(graph_size: int = 10) -> nx.DiGraph:
    """Create a plausible causal graph for agriculture domain."""
    G = nx.DiGraph()

    # Define plausible edges
    # Weather affects soil
    if graph_size >= 6:
        G.add_edges_from([
            ('temperature', 'soil_moisture'),
            ('precipitation', 'soil_moisture'),
            ('precipitation', 'soil_nitrogen'),
        ])

    # Weather affects crop directly
    if graph_size >= 8:
        G.add_edges_from([
            ('temperature', 'crop_growth'),
            ('precipitation', 'crop_growth'),
            ('solar_radiation', 'crop_growth'),
        ])

    # Soil affects crop
    if graph_size >= 5:
        G.add_edges_from([
            ('soil_moisture', 'crop_growth'),
            ('soil_nitrogen', 'crop_growth'),
        ])

    # Management affects crop
    if graph_size >= 7:
        G.add_edges_from([
            ('irrigation', 'crop_growth'),
            ('fertilizer', 'crop_growth'),
            ('fertilizer', 'soil_nitrogen'),
        ])

    return G


def simulate_agriculture_data(n_samples: int = 1000,
                             n_features: int = 10,
                             noise_level: float = 0.1,
                             seed: int = 42) -> Tuple[pd.DataFrame, pd.Series, nx.DiGraph]:
    """Simulate agriculture data with known causal structure."""
    np.random.seed(seed)

    # Define causal structure
    G = create_agriculture_causal_prior(n_features)

    # Sample in topological order
    if len(G.nodes()) > 0:
        order = list(nx.topological_sort(G))
    else:
        order = list(range(n_features))

    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    if len(G.nodes()) == 0:
        # Simple random graph
        nx.add_path(G, feature_names)
        order = feature_names

    # Simulate data according to causal structure
    data = {}

    for var in order:
        if var in G:
            parents = list(G.predecessors(var))
            if len(parents) == 0:
                # Root variable
                data[var] = np.random.normal(0, 1, n_samples)
            else:
                # Regress on parents
                coefs = np.random.uniform(-0.5, 0.5, len(parents))
                parent_values = np.column_stack([data[p] for p in parents])
                mean_val = parent_values @ coefs
                data[var] = mean_val + np.random.normal(0, noise_level, n_samples)

    # Create target (last variable or combination)
    if 'crop_growth' in data:
        target_name = 'crop_growth'
    elif 'yield' in data:
        target_name = 'yield'
    else:
        target_name = order[-1]

    # Generate target
    if target_name in data:
        y = data.pop(target_name) + np.random.normal(0, 0.1, n_samples)
    else:
        # Generate target from features
        y = np.sum(list(data.values()), axis=0) + np.random.normal(0, 0.1, n_samples)

    X = pd.DataFrame(data)
    y = pd.Series(y, name='target')

    return X, y, G


def aggregate_causal_attributions(attributions: pd.DataFrame,
                                  feature_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """Aggregate attributions by feature groups."""
    group_attrs = pd.DataFrame(index=attributions.index)

    for group_name, features in feature_groups.items():
        # Find matching features
        matching_features = [f for f in features if f in attributions.columns]
        if matching_features:
            group_attrs[group_name] = attributions[matching_features].sum(axis=1)

    return group_attrs


def plot_group_attributions(group_attributions: pd.DataFrame,
                           figsize: tuple = (10, 6),
                           top_k: Optional[int] = None) -> None:
    """Plot grouped attributions."""
    import matplotlib.pyplot as plt

    # Compute mean absolute attribution per group
    mean_attrs = group_attributions.abs().mean().sort_values(ascending=True)

    if top_k is not None:
        mean_attrs = mean_attrs.tail(top_k)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['green' if x > 0 else 'red'
              for x in mean_attrs.values]
    ax.barh(range(len(mean_attrs)), mean_attrs.values,
            color=colors, alpha=0.7)

    ax.set_yticks(range(len(mean_attrs)))
    ax.set_yticklabels(mean_attrs.index)
    ax.set_xlabel('Mean Absolute Attribution', fontsize=12)
    ax.set_title('Group-Level Attributions', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def explain_agriculture_model(analyzer, attributions: pd.DataFrame,
                              feature_groups: Optional[Dict[str, List[str]]] = None) -> None:
    """Provide natural language explanation for agriculture model."""
    if feature_groups is None:
        feature_groups = get_agriculture_context()

    print("=" * 60)
    print("CAUSAL ATTRIBUTION ANALYSIS - AGRICULTURE MODEL")
    print("=" * 60)

    # Show causal structure
    print("\nDiscovered Causal Structure:")
    print(f"Number of causal links: {analyzer.causal_graph.number_of_edges()}")
    print(f"Features in causal order: {', '.join(analyzer.causal_order)}")

    # Show top contributing features
    mean_abs_attrs = attributions.abs().mean().sort_values(ascending=False)
    print("\nTop Contributing Features (Average Attribution):")
    for i, (feature, value) in enumerate(mean_abs_attrs.head(5).items(), 1):
        print(f"{i}. {feature}: {value:.4f}")

    # Show group-level contributions if applicable
    group_attrs = aggregate_causal_attributions(attributions, feature_groups)
    if not group_attrs.empty:
        print("\nGroup-Level Contributions:")
        mean_group_attrs = group_attrs.abs().mean().sort_values(ascending=False)
        for group, value in mean_group_attrs.items():
            print(f"- {group}: {value:.4f}")

    print("\n" + "=" * 60)

