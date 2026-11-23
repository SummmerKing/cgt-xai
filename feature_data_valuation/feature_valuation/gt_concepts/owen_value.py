import numpy as np
import itertools
from typing import List
import logging
import math
from .least_core import ValuationResult
from itertools import chain, combinations
import random
from models.train_model import get_model


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
        self.call_count = 0
        self.model_name = model_name
        self.scorer = scorer
        self.data = DataWrapper(list(range(X_train.shape[1])), feature_names)
        self.random_state = random_state
        self._cache = {}

    def __call__(self, subset):
        key = tuple(sorted(subset))
        if key in self._cache:
            return self._cache[key]
        self.call_count +=1
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


class GroupUtility:
    def __init__(self, original_utility, groups):
        self.original_utility = original_utility
        self.groups = groups
        self.data = DataWrapper(list(range(len(groups))))

    def __call__(self, group_subset):
        feature_indices = [
            feature_idx
            for group_idx in group_subset
            for feature_idx in self.groups[group_idx]
        ]
        return self.original_utility(feature_indices)


logger = logging.getLogger(__name__)

def exact_shapley_value(utility, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)
    shapley_values = np.zeros(n)
    factorial_n = math.factorial(n)

    players = range(n)

    for i in players:
        contribution = 0.0
        for S in powerset([p for p in players if p != i]):
            S = list(S)
            weight = (math.factorial(len(S)) *
                      math.factorial(n - len(S) - 1)) / factorial_n
            marginal = utility(S + [i]) - utility(S)
            contribution += weight * marginal
        shapley_values[i] = contribution

    return ValuationResult(
        algorithm="exact_shapley_value",
        status="Converged",
        values=shapley_values,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )

def powerset(iterable):


    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))

def montecarlo_shapley_value(utility, n_iterations=1000, options=None, progress=False):

    if options is None:
        options = {}

    n = len(utility.data)
    shapley_values = np.zeros(n)

    players = np.arange(n)

    for _ in range(n_iterations):
        perm = np.random.permutation(players)
        current_coalition = []
        prev_value = utility(current_coalition)

        for p in perm:
            current_value = utility(current_coalition + [p])
            marginal = current_value - prev_value
            shapley_values[p] += marginal
            current_coalition.append(p)
            prev_value = current_value

    shapley_values /= n_iterations

    return ValuationResult(
        algorithm="montecarlo_shapley_value",
        status="Converged",
        values=shapley_values,
        subsidy=None,
        stderr=None,
        data_names=utility.data.data_names,
    )


def exact_owen_value(utility, groups):

    n_features = len(utility.data)
    owen_values = np.zeros(n_features)

    group_utility = GroupUtility(utility, groups)
    group_shapley_values = exact_shapley_value(group_utility).values

    for i, group in enumerate(groups):
        if not group: continue
        group_total_value = group_shapley_values[i]

        internal_utility = Utility(
            utility.X_train, utility.y_train, utility.X_test, utility.y_test,
            utility.model_name, utility.scorer,
            [utility.data.data_names[j] for j in group], utility.random_state
        )
        internal_utility.data.indices = list(range(len(group)))

        def internal_call(subset_local_indices):
            subset_original_indices = [group[j] for j in subset_local_indices]
            return utility(subset_original_indices)
        internal_utility.__call__ = internal_call

        internal_values = exact_shapley_value(internal_utility).values
        internal_sum = np.sum(internal_values)

        if internal_sum != 0:
            normalized_values = (internal_values / internal_sum) * group_total_value
        else:
            normalized_values = np.full(len(group), group_total_value / len(group))

        for j, original_idx in enumerate(group):
            owen_values[original_idx] = normalized_values[j]

    return ValuationResult(
    algorithm="exact_owen_value",
    status="Converged",
    values=owen_values,
    subsidy=None,
    stderr=None,
    data_names=utility.data.data_names,
)


def montecarlo_owen_value(utility, groups, n_iterations=1000):
    n = len(utility.data)
    owen_values = np.zeros(n)
    group_indices = list(range(len(groups)))

    for _ in range(n_iterations):
        random.shuffle(group_indices)
        perm = []
        for group_idx in group_indices:
            group = groups[group_idx].copy()
            random.shuffle(group)
            perm.extend(group)

        current_coalition = []
        prev_value = utility(current_coalition)
        for p in perm:
            current_value = utility(current_coalition + [p])
            marginal = current_value - prev_value
            owen_values[p] += marginal
            current_coalition.append(p)
            prev_value = current_value

    owen_values /= n_iterations
    return ValuationResult(
    algorithm="montecarlo_owen_value",
    status="Converged",
    values=owen_values,
    subsidy=None,
    stderr=None,
    data_names=utility.data.data_names,
)
