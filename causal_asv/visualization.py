"""Plots for causal graphs and attributions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Optional, List, Dict


def visualize_causal_graph(causal_graph: nx.DiGraph,
                          layout: str = 'spring',
                          figsize: tuple = (12, 8),
                          node_size: int = 1500,
                          node_color: str = 'lightblue',
                          edge_color: str = 'gray',
                          font_size: int = 10,
                          title: str = 'Causal Graph',
                          ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Visualize causal DAG."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Choose layout
    try:
        if layout == 'hierarchical':
            import pygraphviz
            pos = nx.nx_agraph.graphviz_layout(causal_graph, prog='dot')
        else:
            pos = None
    except:
        pos = None

    if pos is None:
        if layout == 'spring':
            pos = nx.spring_layout(causal_graph, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(causal_graph)
        else:
            pos = nx.spring_layout(causal_graph)

    # Draw nodes
    nx.draw_networkx_nodes(
        causal_graph, pos,
        node_size=node_size,
        node_color=node_color,
        alpha=0.8,
        ax=ax
    )

    # Draw edges with arrows
    nx.draw_networkx_edges(
        causal_graph, pos,
        edge_color=edge_color,
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        alpha=0.6,
        width=2,
        ax=ax
    )

    # Draw labels
    nx.draw_networkx_labels(
        causal_graph, pos,
        font_size=font_size,
        font_weight='bold',
        ax=ax
    )

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    if ax is None:
        plt.tight_layout()
    return fig


def plot_attributions(attributions: pd.DataFrame,
                     instance_idx: int = 0,
                     top_k: Optional[int] = None,
                     figsize: tuple = (10, 6),
                     title: str = 'Feature Attributions',
                     ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot feature attributions for a single instance."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Extract instance attributions
    instance_attrs = attributions.iloc[instance_idx]

    # Sort by absolute value
    sorted_attrs = instance_attrs.abs().sort_values(ascending=True)

    # Select top k if specified
    if top_k is not None:
        sorted_attrs = sorted_attrs.tail(top_k)

    # Create horizontal bar plot
    colors = ['red' if x < 0 else 'green' for x in instance_attrs[sorted_attrs.index]]
    bars = ax.barh(
        range(len(sorted_attrs)),
        instance_attrs[sorted_attrs.index],
        color=colors,
        alpha=0.7
    )

    ax.set_yticks(range(len(sorted_attrs)))
    ax.set_yticklabels(sorted_attrs.index)
    ax.set_xlabel('Attribution Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # Add value labels on bars
    for i, (idx, val) in enumerate(instance_attrs[sorted_attrs.index].items()):
        ax.text(val, i, f'{val:.3f}', va='center',
                ha='left' if val > 0 else 'right')

    if ax is None:
        plt.tight_layout()
    return fig


def plot_attribution_summary(attributions: pd.DataFrame,
                            figsize: tuple = (12, 6),
                            title: str = 'Attribution Summary',
                            ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot summary statistics of attributions across all instances."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Compute mean absolute attributions
    mean_abs_attrs = attributions.abs().mean().sort_values(ascending=True)

    # Create horizontal bar plot
    ax.barh(range(len(mean_abs_attrs)), mean_abs_attrs.values,
            color='steelblue', alpha=0.7)

    ax.set_yticks(range(len(mean_abs_attrs)))
    ax.set_yticklabels(mean_abs_attrs.index)
    ax.set_xlabel('Mean Absolute Attribution', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if ax is None:
        plt.tight_layout()
    return fig


def heatmap_attributions(attributions: pd.DataFrame,
                        top_k: Optional[int] = None,
                        figsize: tuple = (12, 8),
                        cmap: str = 'RdBu_r') -> plt.Figure:
    """Create heatmap of attributions across instances."""
    fig, ax = plt.subplots(figsize=figsize)

    # Select top k features if specified
    if top_k is not None:
        # Get features with highest mean absolute attribution
        top_features = attributions.abs().mean().nlargest(top_k).index
        attributions = attributions[top_features]

    # Create heatmap
    sns.heatmap(
        attributions.T,
        cmap=cmap,
        center=0,
        vmin=attributions.min().min(),
        vmax=attributions.max().max(),
        cbar_kws={'label': 'Attribution Value'},
        ax=ax
    )

    ax.set_title('Attribution Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    plt.tight_layout()
    return fig
