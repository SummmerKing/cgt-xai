# utils/data_loader.py
import numpy as np
import os


def load_dataset(name: str) -> Dataset:
    """Load a preprocessed dataset from npz files."""
    if name.lower() == "breast_cancer":
        path = Path("datasets/breast_cancer/breast_cancer_preprocessed.npz")
    elif name.lower() in ["congress", "house"]:
        path = Path("datasets/congress/congress_preprocessed.npz")
    elif name.lower() == "wine":
        path = Path("datasets/wine/wine_preprocessed.npz")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    logger.info(f"Loading dataset from {path}")
    data = np.load(path, allow_pickle=True)  # Added allow_pickle=True
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # First try to get feature names from the NPZ file
    feature_names = None
    if "feature_names" in data:
        feature_names = data["feature_names"].tolist()  # Convert numpy array to list if needed
        logger.info(f"Loaded {len(feature_names)} feature names from NPZ file")

    # If not in NPZ, try text file as fallback
    if feature_names is None:
        feature_names_path = path.parent / f"{name.lower()}_feature_names.txt"
        if feature_names_path.exists():
            with open(feature_names_path, "r") as f:
                feature_names = [line.strip() for line in f if line.strip()]

    return Dataset(X_train, y_train, X_test, y_test, feature_names)
