"""Global feature attribution on the Pima diabetes data with five solution concepts.

The game: v(S) is the test accuracy of a logistic regression trained on feature subset S, minus the
accuracy of predicting the majority class. Every concept splits v(N) among the features.

    python examples/feature_attribution.py
"""
import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cgt_xai.concepts.banzhaf import exact_banzhaf_index
from cgt_xai.concepts.gately_point import exact_gately_value
from cgt_xai.concepts.least_core import exact_least_core
from cgt_xai.concepts.nucleolus import exact_nucleolus
from cgt_xai.concepts.shapley import exact_shapley_value
from cgt_xai.models import get_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"preg": "pregnancies", "plas": "glucose", "pres": "blood pressure", "skin": "skin fold",
         "insu": "insulin", "mass": "BMI", "pedi": "pedigree", "age": "age"}


class FeatureGame:
    def __init__(self, X_train, y_train, X_test, y_test, feature_names):
        self.Xtr, self.ytr, self.Xte, self.yte = X_train, y_train, X_test, y_test
        self.data = type("Players", (), {
            "indices": np.arange(X_train.shape[1]),
            "data_names": list(feature_names),
            "__len__": lambda s: X_train.shape[1],
        })()
        majority = np.bincount(y_train.astype(int)).argmax()
        self.base = float(np.mean(y_test == majority))

    def __call__(self, S):
        return self._value(tuple(sorted(int(i) for i in S)))

    @lru_cache(maxsize=None)
    def _value(self, S):
        if not S:
            return 0.0
        model = get_model("logistic").fit(self.Xtr[:, S], self.ytr)
        return float(model.score(self.Xte[:, S], self.yte)) - self.base


def main():
    d = np.load(os.path.join(ROOT, "datasets", "pima_diabetes_preprocessed.npz"), allow_pickle=True)
    names = [NAMES.get(str(n), str(n)) for n in d["feature_names"]]
    game = FeatureGame(d["X_train"], d["y_train"], d["X_test"], d["y_test"], names)

    results = {
        "Shapley": exact_shapley_value(game).values,
        "Banzhaf": exact_banzhaf_index(game, normalized=True).values * game(range(len(names))),
        "Least core": exact_least_core(game).values,
        "Nucleolus": exact_nucleolus(game).values,
        "Gately": exact_gately_value(game).values,
    }
    table = pd.DataFrame(results, index=names)
    print(f"v(N) = {game(range(len(names))):.3f} accuracy above the majority-class baseline\n")
    print(table.round(4).to_string())

    order = table["Shapley"].sort_values(ascending=False).index
    colors = ["#2a78d6", "#eb6834", "#1f9e89", "#7a5195", "#8c8c8c"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.16
    x = np.arange(len(order))
    for k, (col, c) in enumerate(zip(table.columns, colors)):
        ax.bar(x + (k - 2) * width, table.loc[order, col], width, label=col, color=c)
    ax.set_xticks(x, order, rotation=0)
    ax.set_ylabel("share of v(N) (accuracy)")
    ax.axhline(0, color="#3a3a3a", linewidth=0.8)
    ax.set_title("Pima diabetes, logistic regression: feature attributions by solution concept")
    ax.legend(ncol=5, frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "pima_feature_attribution.png")
    fig.savefig(out, dpi=160)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
