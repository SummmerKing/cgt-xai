# utils/evaluation.py
from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics for the model."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "report": classification_report(y_test, y_pred)
    }
    return metrics
