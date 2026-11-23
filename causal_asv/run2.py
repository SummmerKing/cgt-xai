"""Run Causal ASV on a .npz dataset with keys X_train, X_test, y_train, y_test, feature_names.

    python3 run2.py path/to/data.npz
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from attribution import CausalASVAnalyzer
from visualization import plot_attributions, visualize_causal_graph
import warnings
import sys
import os
import glob

warnings.filterwarnings('ignore')

def main(dataset_path=None):
    print("="*70)
    print("Causal ASV Analysis on NPY/NPZ Dataset")
    print("="*70)

    # Load data - try .npz first, then individual .npy files
    print("\nLoading data...")
    X_train = X_test = y_train = y_test = feature_names = None

    # Determine which .npz file to use
    npz_file = None
    if dataset_path and os.path.exists(dataset_path):
        npz_file = dataset_path
        print(f"Using specified file: {npz_file}")
    else:
        # Try to load from a single .npz file in current directory
        npz_files = glob.glob('*.npz')
        if npz_files:
            npz_file = npz_files[0]
            print(f"Found .npz file: {npz_file}")

    if npz_file:
        data = np.load(npz_file)
        print(f"Available keys in .npz: {list(data.keys())}")

        # Load from .npz file
        if 'X_train' in data:
            X_train = data['X_train']
        if 'X_test' in data:
            X_test = data['X_test']
        if 'y_train' in data:
            y_train = data['y_train']
        if 'y_test' in data:
            y_test = data['y_test']
        if 'feature_names' in data:
            feature_names = data['feature_names']

    # If not found in .npz, try individual .npy files
    try:
        if X_train is None:
            X_train = np.load('X_train.npy')
        if X_test is None:
            X_test = np.load('X_test.npy')
        if y_train is None:
            y_train = np.load('y_train.npy')
        if y_test is None:
            y_test = np.load('y_test.npy')
        if feature_names is None:
            feature_names = np.load('feature_names.npy')
    except FileNotFoundError as e:
        print(f"\nError: Could not find data files")
        print("Please ensure you have one of:")
        print("  1. A single .npz file with keys: X_train, X_test, y_train, y_test, feature_names")
        print("  2. Individual .npy files: X_train.npy, X_test.npy, y_train.npy, y_test.npy, feature_names.npy")
        print(f"\nMissing: {e}")
        return

    print(f"Loaded training data: {X_train.shape}")
    print(f"Loaded test data: {X_test.shape}")
    print(f"Feature names: {list(feature_names)}")

    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_train_series = pd.Series(y_train, name='target')
    y_test_series = pd.Series(y_test, name='target')

    # Train model
    print("\n" + "="*70)
    print("Training Model")
    print("="*70)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_df, y_train_series)

    # Check if classification or regression
    is_regression = y_train_series.dtype in ['float64', 'float32', 'float'] and y_train_series.nunique() > 2

    def model_predict(X):
        if hasattr(model, 'predict_proba') and not is_regression:
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)

    accuracy = model.score(X_test_df, y_test_series)
    if is_regression:
        print(f"Model R² score: {accuracy:.3f}")
    else:
        print(f"Model accuracy: {accuracy:.3f}")

    # Discover causal structure
    print("\n" + "="*70)
    print("Discovering Causal Structure")
    print("="*70)
    analyzer = CausalASVAnalyzer(
        model=model_predict,
        significance_level=0.05
    )

    causal_graph = analyzer.discover_causal_structure(X_train_df, verbose=True)

    print(f"\nDiscovered {causal_graph.number_of_edges()} causal links")
    print(f"Causal order: {analyzer.get_causal_order()}")

    # Compute attributions
    print("\n" + "="*70)
    print("Computing Asymmetric Shapley Values")
    print("="*70)

    # Explain a subset of test instances (adjust as needed)
    n_explain = min(10, len(X_test_df))
    print(f"Computing attributions for {n_explain} test instances...")

    attributions = analyzer.shapley_values(
        X_test_df.head(n_explain),
        max_samples=100
    )

    print(f"Computed attributions: {attributions.shape}")

    # Create visualizations
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)

    # Plot attributions for first instance
    plot_attributions(attributions, instance_idx=0, top_k=4)

    # Visualize causal graph
    visualize_causal_graph(causal_graph, title="Discovered Causal Structure")

    # Summary
    print("\n" + "="*70)
    print("Attribution Summary")
    print("="*70)
    mean_abs_attrs = attributions.abs().mean().sort_values(ascending=False)

    print("\nMean Absolute Attribution by Feature:")
    print("-" * 70)
    for feat, val in mean_abs_attrs.items():
        print(f"  {feat:20s}: {val:.4f}")

    print("\n" + "="*70)
    print("Top Contributing Features (Average Attribution)")
    print("="*70)
    for i, (feat, val) in enumerate(mean_abs_attrs.items(), 1):
        print(f"{i}. {feat:20s}: {val:.4f}")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

if __name__ == "__main__":
    # Allow command-line argument for dataset path
    dataset_path = None
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    main(dataset_path=dataset_path)

