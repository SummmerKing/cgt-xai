import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def plot_data_valuation(valuation_results, dataset_name, output_dir=None):
    """Plot the valuation results."""
    sns.set_style("whitegrid")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Create figure
    fig, axes = plt.subplots(1, len(valuation_results), figsize=(5*len(valuation_results), 5))

    if len(valuation_results) == 1:
        axes = [axes]

    # Plot each method's values
    for i, (method, result) in enumerate(valuation_results.items()):
        values = result.values
        ax = axes[i]

        # Plot histogram
        ax.hist(values, bins=30, alpha=0.7, color=colors[i % len(colors)])
        ax.set_title(f"{method}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    plt.tight_layout()

    # Save figure if output_dir specified
    if output_dir:
        output_path = f"{output_dir}/{dataset_name}_valuation_histograms.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig

def plot_removal_experiment(best_removal_results, worst_removal_results, removal_percentages, dataset_name, output_dir=None):
    """Plot the results of the removal experiment."""
    # Set style
    sns.set_style("whitegrid")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot best removal results
    for i, (method, scores) in enumerate(best_removal_results.items()):
        ax1.plot(removal_percentages, scores, 'o-', label=method, color=colors[i % len(colors)])

    ax1.set_title(f"{dataset_name}: Remove Best Points")
    ax1.set_xlabel("Fraction of points removed")
    ax1.set_ylabel("Test Accuracy")
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(bottom=0.5)  # Adjust as needed
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Plot worst removal results
    for i, (method, scores) in enumerate(worst_removal_results.items()):
        ax2.plot(removal_percentages, scores, 'o-', label=method, color=colors[i % len(colors)])

    ax2.set_title(f"{dataset_name}: Remove Worst Points")
    ax2.set_xlabel("Fraction of points removed")
    ax2.set_ylabel("Test Accuracy")
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(bottom=0.5)  # Adjust as needed
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()

    # Save figure if output_dir specified
    if output_dir:
        output_path = f"{output_dir}/{dataset_name}_removal_experiment.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig
