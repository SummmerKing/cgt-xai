"""Example: Causal ASV attribution on a CSV dataset."""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils import load_data, simulate_agriculture_data
from attribution import CausalASVAnalyzer
from visualization import plot_attributions, visualize_causal_graph
import numpy as np

def main():
    filepath = "../datasets/cv_risk.csv"
    try:
        X, y = load_data(filepath)
    except Exception as e:
        print(f"Could not load data from {filepath}: {e}")
        print("Using simulated data instead...")
        X, y, true_graph = simulate_agriculture_data(n_samples=200, n_features=8)

    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {list(X.columns)}")

    # Step 2: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Train a model (auto-detect regression vs classification)
    print("\nTraining model...")
    is_regression = y_train.dtype in ['float64', 'float32', 'float', 'int64', 'int32'] and y_train.nunique() > 2

    if is_regression:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        print("Using RandomForestRegressor for continuous target")
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Using RandomForestClassifier for discrete target")

    model.fit(X_train, y_train)

    # Create a wrapper for model prediction
    def model_predict(X):
        if hasattr(model, 'predict_proba') and not is_regression:
            # Classification case
            return model.predict_proba(X)[:, 1]
        else:
            # Regression case
            return model.predict(X)

    score = model.score(X_test, y_test)
    if is_regression:
        print(f"Model R² score: {score:.3f}")
    else:
        print(f"Model accuracy: {score:.3f}")

    # Step 4: Initialize Causal ASV Analyzer
    print("\nDiscovering causal structure...")
    analyzer = CausalASVAnalyzer(
        model=model_predict,
        significance_level=0.05
    )

    # Discover causal structure from training data
    causal_graph = analyzer.discover_causal_structure(X_train, verbose=True)

    print(f"\nDiscovered {causal_graph.number_of_edges()} causal links")
    print(f"Causal order: {analyzer.get_causal_order()}")

    # Step 5: Compute attributions
    print("\nComputing Asymmetric Shapley Values...")
    attributions = analyzer.shapley_values(
        X_test.head(10),  # Explain first 10 test instances
        max_samples=100  # Monte Carlo samples per feature
    )

    print(f"Computed attributions for {len(attributions)} instances")

    # Step 6: Visualize results
    print("\nCreating visualizations...")

    # Plot attributions for first instance
    plot_attributions(attributions, instance_idx=0, top_k=5)

    # Visualize causal graph
    visualize_causal_graph(causal_graph, title="Discovered Causal Structure")

    # Summary
    print("\n" + "="*60)
    print("Attribution Summary (Top 5 features by absolute value):")
    print("="*60)
    mean_attrs = attributions.abs().mean().sort_values(ascending=False)
    for i, (feat, val) in enumerate(mean_attrs.head(5).items(), 1):
        print(f"{i}. {feat}: {val:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
