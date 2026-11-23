# Cooperative Game Theory for Explainable AI

Feature attribution and data valuation with cooperative game theory. A model's performance on a subset of features (or training points) defines a coalitional game, and different solution concepts of that game give different explanations. The project compares these concepts against each other and against causal variants of Shapley values, and evaluates them with faithfulness and robustness metrics.

Joint work at the Game Theory Lab, IISc. An early version was presented at the Foundations of Ethical AI workshop (IIIT Hyderabad).

## Solution concepts

`gt_concepts/` implements exact and Monte Carlo versions of:

- Shapley value, Banzhaf index, Shapley-Shubik and Deegan-Packel power indices
- least core and nucleolus (LP/QP based)
- Gately point
- data valuation baselines: leave-one-out, TMC-Shapley, group-testing Shapley
- Causal SHAP (`gt_concepts/causal_shap.py`): PC causal discovery, DAG sampling from the CPDAG, linear mechanisms and interventional Shapley values

`causal_asv/` implements asymmetric Shapley values that follow a causal ordering found by the PC algorithm. The Owen and Myerson values are in `CGT_Explainability/feature_valuation/gt_concepts/` and `lung_cv_risk/`.

## Experiments

| Folder | What it covers |
| --- | --- |
| `local_explanation/` | local explanations with loss-based and conditional-expectation utilities on wine, soil, yield, COMPAS, credit and diabetes data |
| `CGT_Explainability/` | global feature valuation (logistic regression, random forest, SVM) and data valuation on UCI datasets |
| `if_you_like_shapley_experiments/` | least core, Gately and Shapley allocations with accuracy and log-loss utilities |
| `metrics/` | faithfulness and robustness (Quantus and custom metrics) for each concept, including LSTM and RNN models |
| `causal_shap_exp/`, `causal_shap_*.ipynb` | Causal SHAP on soil and yield data, graph and edge-weight sensitivity, robustness |
| `causal_dagma/`, `causal_bayesian_op/`, `causal_alias/` | Causal SHAP with graphs from DAGMA / DAGMA-DCE, DrBO and ALIAS |
| `asymmetric_shap/`, `causal_asv/` | asymmetric Shapley values on lung cancer and cardiovascular risk data |
| `lung_cv_risk/`, `soil_dataset/`, `soil_accuracy_scalability/` | per-dataset comparisons of the solution concepts |
| `myerson_value/` | Myerson value with a feature graph |

Preprocessed datasets are in `datasets/` and `CGT_Explainability/datasets/` as `.npz` files. Raw CSV files (cardiovascular and lung cancer risk, FAO data) are not included.

## Setup

```bash
pip install -r requirements.txt
```

The causal discovery notebooks also use two external packages:

```bash
# DAGMA-DCE
git clone https://github.com/DanWaxman/DAGMA-DCE && pip install -e DAGMA-DCE/src

# DrBO (run from causal_bayesian_op/)
./setup_drbo.sh
```

Notebooks add the repository root to `sys.path` to import the shared modules (`gt_concepts`, `models`, `utils`), and data paths are relative to each notebook's folder, so run them from their own folder (the Jupyter default).
