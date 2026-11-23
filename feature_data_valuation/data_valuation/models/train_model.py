import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_model(model_name: str = "logistic", random_state: int = 42):

    if model_name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=random_state, max_iter=1000))
        ])
    elif model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state)
    elif model_name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=random_state))
        ])
