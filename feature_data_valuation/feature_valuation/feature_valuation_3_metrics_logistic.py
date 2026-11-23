import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from scipy.stats import spearmanr, pearsonr
from numpy.linalg import norm
import matplotlib.pyplot as plt

from gt_concepts.shapley import exact_shapley_value, montecarlo_shapley_value
from gt_concepts.least_core import exact_least_core, montecarlo_least_core, ValuationResult
from gt_concepts.nucleolus import montecarlo_nucleolus, exact_nucleolus
from gt_concepts.banzhaf import exact_banzhaf_index, montecarlo_banzhaf_index
from gt_concepts.shapley_shubik import exact_shapley_shubik, montecarlo_shapley_shubik
from gt_concepts.gately_point import exact_gately_value, montecarlo_gately_value
from gt_concepts.deegan_packel import exact_deegan_packel_index, montecarlo_deegan_packel_index
from models.train_model import get_model
from utils.plotting import plot_single_valuation
from utils.evaluation import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "breast_cancer"
MODEL_NAME = "logistic"
SCORING = "accuracy"
METHODS = ["shapley","least_core","nucleolus","banzhaf", "shapley_shubik","gately" ]
MONTE_CARLO = False
N_MC_ITERATIONS = 10000
RANDOM_STATE = 42

OUTPUT_DIR = Path(f"./output/feature_valuation/{DATASET_NAME}_{MODEL_NAME}_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset(name: str):
    path_map = {
        "breast_cancer": "../datasets/breast_cancer/breast_cancer_preprocessed.npz",
        "congress": "../datasets/congress/congress_preprocessed.npz",
        "wine": "../datasets/wine/wine_preprocessed.npz",
        "apple_quality": "../datasets/apple_quality/apple_quality_preprocessed_no_scaling.npz",
        "yield_synthetic": "../datasets/yield_synthetic/yield_preprocessed.npz"
    }
    path = Path(path_map[name])
    data = np.load(path, allow_pickle=True)
    feature_names = data["feature_names"].tolist() if "feature_names" in data else [f"f{i}" for i in range(data["X_train"].shape[1])]
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"], feature_names

class DataWrapper:
    def __init__(self, indices, names=None):
        self.indices = indices
        self.data_names = names if names is not None else [f"f{i}" for i in range(len(indices))]

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

class Utility:
    def __init__(self, X_train, y_train, X_test, y_test, model_name, scorer, feature_names=None, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.scorer = scorer
        self.data = DataWrapper(list(range(X_train.shape[1])), feature_names)
        self.random_state = random_state
        self._cache = {}

    def __call__(self, subset):
        key = tuple(sorted(subset))
        if key in self._cache:
            return self._cache[key]
        if len(key) == 0:
            majority = np.bincount(self.y_train.astype(int)).argmax()
            y_pred = np.full_like(self.y_test, majority)
            score = self.scorer(self.y_test, y_pred)
            self._cache[key] = score
            return score
        model = get_model(self.model_name, self.random_state)
        model.fit(self.X_train[:, key], self.y_train)
        y_pred = model.predict(self.X_test[:, key])
        score = self.scorer(self.y_test, y_pred)
        self._cache[key] = score
        return score


from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error

def get_scorer(name):
    if name == "accuracy":
        return lambda y_true, y_pred: accuracy_score(y_true, y_pred)
    elif name == "f1_macro":
        return lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")
    elif name == "r2":
        return lambda y_true, y_pred: r2_score(y_true, y_pred)
    elif name == "mse":
        return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    elif name == "rmse":
        return lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))
    elif name == "mae":
        return lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError(f"Unknown scorer: {name}")

X_train, X_test, y_train, y_test, feature_names = load_dataset(DATASET_NAME)
scorer = get_scorer(SCORING)
utility = Utility(
    X_train, y_train, X_test, y_test,
    MODEL_NAME,
    scorer,
    feature_names=feature_names,
    random_state=RANDOM_STATE
)

n_features = X_train.shape[1]
full_model_accuracy = utility(list(range(n_features)))
SHAPLEY_SHUBIK_THRESHOLD = full_model_accuracy / (1* n_features)

print(full_model_accuracy)
print(SHAPLEY_SHUBIK_THRESHOLD)


df = pd.read_csv("feature_valuation/output/feature_valuation/breast_cancer_logistic_results/breast_cancer_logistic_feature_valuations.csv")
results = {}
for method in df["method"].unique():
    vals = df[df["method"] == method]["value"].values
    results[method] = vals


logger.info("\n" + "="*70)
logger.info("EVALUATING GLOBAL FAITHFULNESS METRICS")
logger.info("="*70)


logger.info("\nCalculating Global Sensitivity (Leave-One-Out Impact)...")

n_features = X_train.shape[1]
all_features_set = set(range(n_features))
v_N = utility(list(all_features_set))

global_impacts = []
for i in range(n_features):
    features_except_i = list(all_features_set - {i})
    v_N_minus_i = utility(features_except_i)
    impact = v_N - v_N_minus_i
    global_impacts.append(impact)
    logger.info(f"  Impact of removing '{feature_names[i]}': {impact:.4f}")

global_impact_vector = np.array(global_impacts)

print(global_impact_vector)

sensitivity_results = []
for method_name, attribution_values in results.items():
    if attribution_values is None:
        logger.warning(f"  Skipping {method_name} (None values)")
        continue

    corr_spearman, p_spearman = spearmanr(attribution_values, global_impact_vector)

    corr_pearson, p_pearson = pearsonr(attribution_values, global_impact_vector)

    cos_sim = np.dot(attribution_values, global_impact_vector) / (
        np.linalg.norm(attribution_values) * np.linalg.norm(global_impact_vector) + 1e-10
    )

    sensitivity_results.append({
        "method": method_name,
        "sensitivity_spearman": corr_spearman,
        "p_value_spearman": p_spearman,
        "sensitivity_pearson": corr_pearson,
        "p_value_pearson": p_pearson,
        "cosine_similarity": cos_sim
    })

    logger.info(f"  {method_name}: Spearman={corr_spearman:.4f}, Pearson={corr_pearson:.4f}, Cosine={cos_sim:.4f}")

sensitivity_df = pd.DataFrame(sensitivity_results)


print("GLOBAL SENSITIVITY (FAITHFULNESS) METRICS")
print(sensitivity_df.to_string(index=False))


sensitivity_df.to_csv(
    OUTPUT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_global_sensitivity.csv",
    index=False
)


noise_level = 0.05
noise = np.random.normal(
    0,
    noise_level * X_train.std(axis=0),
    X_train.shape
)
X_train_noisy = X_train + noise


utility_noisy = Utility(
    X_train_noisy, y_train, X_test, y_test,
    MODEL_NAME,
    scorer,
    feature_names=feature_names,
    random_state=RANDOM_STATE
)


noisy_results = {}
use_exact = not MONTE_CARLO

if "shapley" in METHODS:
    val = exact_shapley_value(utility_noisy) if use_exact else montecarlo_shapley_value(utility_noisy, N_MC_ITERATIONS)
    noisy_results["shapley"] = val.values

if "least_core" in METHODS:
    val = exact_least_core(utility_noisy) if use_exact else montecarlo_least_core(utility_noisy, N_MC_ITERATIONS)
    noisy_results["least_core"] = val.values

if "nucleolus" in METHODS:
    val = exact_nucleolus(utility_noisy, N_MC_ITERATIONS) if use_exact else montecarlo_nucleolus(utility_noisy, N_MC_ITERATIONS)
    noisy_results["nucleolus"] = val.values if val is not None else None

if "banzhaf" in METHODS:
    val = exact_banzhaf_index(utility_noisy) if use_exact else montecarlo_banzhaf_index(utility_noisy, N_MC_ITERATIONS, normalized=True)
    noisy_results["banzhaf"] = val.values

if "shapley_shubik" in METHODS:
    full_acc_noisy = utility_noisy(list(range(n_features)))
    threshold_noisy = full_acc_noisy / n_features
    val = exact_shapley_shubik(utility_noisy, threshold_noisy) if use_exact else montecarlo_shapley_shubik(utility_noisy, N_MC_ITERATIONS)
    noisy_results["shapley_shubik"] = val.values

if "gately" in METHODS:
    val = exact_gately_value(utility_noisy) if use_exact else montecarlo_gately_value(utility_noisy, N_MC_ITERATIONS)
    noisy_results["gately"] = val.values

if "deegan_packel" in METHODS:
    full_acc_noisy = utility_noisy(list(range(n_features)))
    val = exact_deegan_packel_index(utility_noisy, full_acc_noisy, normalized=True) if use_exact else montecarlo_deegan_packel_index(utility_noisy, N_MC_ITERATIONS, normalized=True)
    noisy_results["deegan_packel"] = val.values


robustness_results = []
for method_name in results.keys():
    if method_name not in noisy_results or noisy_results[method_name] is None or results[method_name] is None:
        logger.warning(f"  Skipping {method_name} (missing data)")
        continue

    phi_original = results[method_name]
    phi_noisy = noisy_results[method_name]

    cos_sim = np.dot(phi_original, phi_noisy) / (
        (np.linalg.norm(phi_original) * np.linalg.norm(phi_noisy)) + 1e-10
    )

    l2_dist = np.linalg.norm(phi_original - phi_noisy)
    l2_norm = l2_dist / (np.linalg.norm(phi_original) + 1e-10)

    rank_corr, _ = spearmanr(phi_original, phi_noisy)

    robustness_results.append({
        "method": method_name,
        "cosine_similarity": cos_sim,
        "rank_correlation": rank_corr,
        "normalized_l2_distance": l2_norm
    })

    logger.info(f"  {method_name}: Cosine={cos_sim:.4f}, RankCorr={rank_corr:.4f}, L2={l2_norm:.4f}")

robustness_df = pd.DataFrame(robustness_results)

print("\n" + "="*70)
print("GLOBAL ROBUSTNESS (STABILITY) METRICS")
print("="*70)
print(robustness_df.to_string(index=False))


robustness_df.to_csv(
    OUTPUT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_global_robustness.csv",
    index=False
)


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax1 = axes[0]
methods = sensitivity_df["method"]
sens_scores = sensitivity_df["sensitivity_spearman"]
colors = ['steelblue' if x > 0.5 else 'coral' for x in sens_scores]

ax1.barh(methods, sens_scores, color=colors)
ax1.set_xlabel("Spearman Correlation", fontsize=12)
ax1.set_title("Global Sensitivity\n(Attribution ↔ Leave-One-Out Impact)", fontsize=14, fontweight='bold')
ax1.axvline(0, color='red', linestyle='--', linewidth=1, label='No correlation')
ax1.axvline(0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Strong correlation')
ax1.grid(axis='x', alpha=0.3)
ax1.legend(fontsize=9)

ax2 = axes[1]
cos_scores = sensitivity_df["cosine_similarity"]
colors2 = ['darkgreen' if x > 0.7 else 'orange' for x in cos_scores]

ax2.barh(methods, cos_scores, color=colors2)
ax2.set_xlabel("Cosine Similarity", fontsize=12)
ax2.set_title("Attribution Alignment\n(Cosine Similarity)", fontsize=14, fontweight='bold')
ax2.axvline(1.0, color='green', linestyle='--', linewidth=1, label='Perfect alignment')
ax2.grid(axis='x', alpha=0.3)
ax2.legend(fontsize=9)

# Plot 3: Global Robustness
ax3 = axes[2]
if len(robustness_df) > 0:
    rob_scores = robustness_df["cosine_similarity"]
    colors3 = ['purple' if x > 0.9 else 'salmon' for x in rob_scores]

    ax3.barh(robustness_df["method"], rob_scores, color=colors3)
    ax3.set_xlabel("Cosine Similarity (Original ↔ Noisy)", fontsize=12)
    ax3.set_title("Global Robustness\n(Stability Under Noise)", fontsize=14, fontweight='bold')
    ax3.axvline(1.0, color='green', linestyle='--', linewidth=1, label='Perfect stability')
    ax3.axvline(0.9, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good stability')
    ax3.grid(axis='x', alpha=0.3)
    ax3.legend(fontsize=9)
else:
    ax3.text(0.5, 0.5, 'Robustness data not available',
             ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title("Global Robustness", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_global_faithfulness.png",
    dpi=300, bbox_inches='tight'
)
plt.show()

logger.info(f"\nAll faithfulness results saved to: {OUTPUT_DIR}")
logger.info("="*70)


