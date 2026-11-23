import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def plot_feature_valuation_comparison(df, feature_names=None, title="Feature Valuation Comparison", output_dir=None):

    methods = df["method"].unique()
    n_methods = len(methods)

    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 5 * n_methods), sharex=False)
    if n_methods == 1:
        axes = [axes]

    for i, method in enumerate(methods):
        method_df = df[df["method"] == method].copy()
        # No filtering by top_k - display all features

        bars = axes[i].barh(method_df["feature"], method_df["value"], color='skyblue')

        # Add values at the end of each bar
        for bar in bars:
            width = bar.get_width()
            axes[i].text(
                width + 0.01 * max(method_df["value"]),
                bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                va='center'
            )

        axes[i].set_title(f"{method}")
        axes[i].set_xlabel("Feature Value")
        axes[i].set_ylabel("Features")
        axes[i].grid(axis="x", linestyle="--", alpha=0.7)

        # Remove top and right spines
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if output_dir:
        output_file = Path(output_dir) / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        logger.info(f"Plot saved to {output_file}")

    return fig, axes

def plot_single_valuation(values, feature_names, method_name="Shapley",
                          title=None, top_k=10, output_path=None):

    df = pd.DataFrame({
        "Feature": feature_names,
        "Value": values
    })


    plt.figure(figsize=(10, 6))
    bars = plt.barh(df["Feature"], df["Value"], color='skyblue')

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01 * max(df["Value"]),
            bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            va='center'
        )

    if title is None:
        title = f"{method_name} Feature Importance"
    plt.title(title, fontsize=14)
    plt.xlabel(f"{method_name} Value (Feature Importance)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        logger.info(f"Plot saved to {output_path}")

    return plt.gcf()


def plot_data_valuation(values, method_name, title=None, output_path=None,
                        top_k=20, bottom_k=20, figsize=(12, 10)):
    """Plot data valuation results showing top_k most valuable and bottom_k least valuable samples."""
    # Sort indices by values
    sorted_indices = np.argsort(values)

    # Get top and bottom indices
    bottom_indices = sorted_indices[:bottom_k]
    top_indices = sorted_indices[-top_k:][::-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot top samples
    ax1.bar(range(len(top_indices)), values[top_indices], color='steelblue')
    ax1.set_xticks(range(len(top_indices)))
    ax1.set_xticklabels([f"Sample {i}" for i in top_indices], rotation=45, ha='right')
    ax1.set_title(f"Top {top_k} Most Valuable Samples")
    ax1.set_ylabel("Sample Value")

    # Plot bottom samples
    ax2.bar(range(len(bottom_indices)), values[bottom_indices], color='indianred')
    ax2.set_xticks(range(len(bottom_indices)))
    ax2.set_xticklabels([f"Sample {i}" for i in bottom_indices], rotation=45, ha='right')
    ax2.set_title(f"Bottom {bottom_k} Least Valuable Samples")
    ax2.set_ylabel("Sample Value")

    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f"Data Valuation with {method_name}", fontsize=16)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)

    return fig, (ax1, ax2)

def plot_removal_experiment(removal_percentages, scores, method_names, title=None,
                           output_path=None, figsize=(10, 6)):
    """Plot model performance as data points are removed according to different valuation methods."""
    plt.figure(figsize=figsize)

    colors = ['steelblue', 'indianred', 'forestgreen', 'darkorange', 'mediumpurple']

    for i, method in enumerate(method_names):
        if method in scores:
            plt.plot(removal_percentages, scores[method],
                     marker='o', label=method, color=colors[i % len(colors)])

    plt.xlabel("Percentage of Data Removed")
    plt.ylabel("Model Performance")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if title:
        plt.title(title)
    else:
        plt.title("Model Performance vs. Data Removal")

    if output_path:
        plt.savefig(output_path)

    return plt.gcf()
