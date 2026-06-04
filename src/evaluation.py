from __future__ import annotations

from collections import Counter
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from models import ModelSpec


def class_distribution(y: pd.Series) -> dict[int, int]:
    return {int(label): int(count) for label, count in y.value_counts().sort_index().items()}


def compute_outer_folds(y: pd.Series, max_outer_folds: int) -> tuple[int, int]:
    counts = y.value_counts()
    n_min = int(counts.min())
    k_outer = min(max_outer_folds, n_min)
    if k_outer < 2:
        raise ValueError(
            "No es posible aplicar validacion cruzada estratificada: "
            "existe una clase con un solo ejemplo."
        )
    return n_min, k_outer


def unimplemented_result(
    target_name: str,
    model_spec: ModelSpec,
    distribution: dict[int, int],
    n_min: int,
    k_outer: int,
) -> dict[str, Any]:
    return {
        "target": target_name,
        "model_key": model_spec.key,
        "model_name": model_spec.display_name,
        "implemented": False,
        "status": "No implementado",
        "message": model_spec.student_note,
        "class_distribution": distribution,
        "n_min": n_min,
        "k_outer": k_outer,
        "k_inner_requested": None,
        "accuracy_mean": None,
        "accuracy_std": None,
        "balanced_accuracy_mean": None,
        "balanced_accuracy_std": None,
        "precision_macro_mean": None,
        "precision_macro_std": None,
        "recall_macro_mean": None,
        "recall_macro_std": None,
        "f1_macro_mean": None,
        "f1_macro_std": None,
        "stability": None,
        "icn": None,
        "best_params_mode": "No implementado",
        "best_params_counts": {},
        "warnings": [],
        "labels": sorted(distribution),
        "confusion_matrix": None,
        "classification_report": None,
    }


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    model_spec: ModelSpec,
    config: dict[str, Any],
) -> dict[str, Any]:
    distribution = class_distribution(y)
    n_min, k_outer = compute_outer_folds(y, config["max_outer_folds"])
    if not model_spec.implemented:
        return unimplemented_result(target_name, model_spec, distribution, n_min, k_outer)

    if model_spec.pipeline is None or model_spec.param_grid is None:
        raise ValueError(f"El modelo {model_spec.key} esta marcado como implementado, pero no tiene pipeline.")

    outer_cv = StratifiedKFold(
        n_splits=k_outer,
        shuffle=True,
        random_state=config["outer_random_state"],
    )
    requested_inner = max(2, min(config["max_inner_folds"], k_outer))
    scorer = make_scorer(f1_score, average="macro", zero_division=0)
    labels = sorted(distribution)

    fold_metrics: list[dict[str, float]] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    best_params_counter: Counter[str] = Counter()
    result_warnings: list[str] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv, inner_warning = _build_inner_cv(
            y_train=y_train,
            requested_inner=requested_inner,
            random_state=config["inner_random_state"] + fold_idx,
            target_name=target_name,
        )
        if inner_warning:
            result_warnings.append(inner_warning)

        search = GridSearchCV(
            estimator=clone(model_spec.pipeline),
            param_grid=model_spec.param_grid,
            scoring=scorer,
            cv=inner_cv,
            n_jobs=config["n_jobs"],
            refit=True,
            error_score="raise",
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            search.fit(X_train, y_train)
            for warning_item in caught_warnings:
                message = str(warning_item.message)
                if message not in result_warnings:
                    result_warnings.append(message)

        y_pred = search.predict(X_test)
        y_test_list = [int(value) for value in y_test.to_list()]
        y_pred_list = [int(value) for value in y_pred.tolist()]

        all_true.extend(y_test_list)
        all_pred.extend(y_pred_list)
        best_params_counter[_format_params(search.best_params_)] += 1

        fold_metrics.append(_compute_fold_metrics(y_test_list, y_pred_list))

    metrics_df = pd.DataFrame(fold_metrics)
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    report = classification_report(
        all_true,
        all_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "target": target_name,
        "model_key": model_spec.key,
        "model_name": model_spec.display_name,
        "implemented": True,
        "status": "Implementado",
        "message": "",
        "class_distribution": distribution,
        "n_min": n_min,
        "k_outer": k_outer,
        "k_inner_requested": requested_inner,
        "best_params_mode": _best_params_mode(best_params_counter),
        "best_params_counts": dict(best_params_counter),
        "warnings": result_warnings,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    for metric_name in metrics_df.columns:
        result[f"{metric_name}_mean"] = float(metrics_df[metric_name].mean())
        result[f"{metric_name}_std"] = float(metrics_df[metric_name].std(ddof=1)) if len(metrics_df) > 1 else 0.0

    result["stability"] = float(max(0.0, min(1.0, 1.0 - result["f1_macro_std"])))
    result["icn"] = None
    return result


def assign_icn(results: list[dict[str, Any]]) -> None:
    implemented = [item for item in results if item["implemented"]]
    if not implemented:
        return

    if len(implemented) == 1:
        item = implemented[0]
        item["icn"] = float(
            0.40 * item["f1_macro_mean"]
            + 0.25 * item["balanced_accuracy_mean"]
            + 0.20 * item["recall_macro_mean"]
            + 0.10 * item["precision_macro_mean"]
            + 0.05 * item["stability"]
        )
        item["icn_note"] = "ICN directo porque solo hay un modelo implementado."
        return

    metric_keys = [
        "f1_macro_mean",
        "balanced_accuracy_mean",
        "recall_macro_mean",
        "precision_macro_mean",
        "f1_macro_std",
    ]
    normalized: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = np.array([item[key] for item in implemented], dtype=float)
        min_value = float(values.min())
        max_value = float(values.max())
        denom = max_value - min_value + 1e-12
        normalized[key] = {}
        for item in implemented:
            if key == "f1_macro_std":
                normalized[key][item["model_key"]] = float(1.0 - (item[key] - min_value) / denom)
            else:
                normalized[key][item["model_key"]] = float((item[key] - min_value) / denom)

    for item in implemented:
        key = item["model_key"]
        item["stability"] = normalized["f1_macro_std"][key]
        item["icn"] = float(
            0.40 * normalized["f1_macro_mean"][key]
            + 0.25 * normalized["balanced_accuracy_mean"][key]
            + 0.20 * normalized["recall_macro_mean"][key]
            + 0.10 * normalized["precision_macro_mean"][key]
            + 0.05 * normalized["f1_macro_std"][key]
        )
        item["icn_note"] = "ICN normalizado min-max entre modelos implementados."


def _build_inner_cv(
    y_train: pd.Series,
    requested_inner: int,
    random_state: int,
    target_name: str,
) -> tuple[StratifiedKFold | KFold, str | None]:
    min_train_count = int(y_train.value_counts().min())
    if min_train_count >= requested_inner:
        return (
            StratifiedKFold(n_splits=requested_inner, shuffle=True, random_state=random_state),
            None,
        )
    if min_train_count >= 2:
        adjusted_inner = min(requested_inner, min_train_count)
        return (
            StratifiedKFold(n_splits=adjusted_inner, shuffle=True, random_state=random_state),
            (
                f"{target_name}: k_inner ajustado de {requested_inner} a {adjusted_inner} "
                f"por soporte minimo {min_train_count} en entrenamiento externo."
            ),
        )

    return (
        KFold(n_splits=2, shuffle=True, random_state=random_state),
        (
            f"{target_name}: ciclo interno usa KFold no estratificado porque una clase tiene "
            "solo 1 ejemplo dentro de un entrenamiento externo."
        ),
    )


def _compute_fold_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _format_params(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _best_params_mode(counter: Counter[str]) -> str:
    if not counter:
        return ""
    value, count = counter.most_common(1)[0]
    total = sum(counter.values())
    return f"{value} ({count}/{total})"
