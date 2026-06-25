from __future__ import annotations

from collections import Counter
from pathlib import Path
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
        "stability_raw": None,
        "icn": None,
        "icn_raw": None,
        "best_params_mode": "No implementado",
        "best_params_counts": {},
        "warnings": [],
        "labels": sorted(distribution),
        "confusion_matrix": None,
        "classification_report": None,
        "best_scores_internal": None,
        "best_score_internal_mean": None,
        "best_score_internal_std": None,
        "fold_metrics": None,
        "fold_f1_external": None,
        "fold_delta_sesgo": None,
        "delta_sesgo": None,
    }


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    model_spec: ModelSpec,
    config: dict[str, Any],
    return_estimators: bool = False,
    estimator_cache_dir: Path | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], list]:
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
    fold_best_scores: list[float] = []
    fold_best_estimators: list = [] if return_estimators else None
    fold_test_indices: list[list[int]] = [] if return_estimators else None

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
                if "fork()" in message or message in result_warnings:
                    continue
                result_warnings.append(message)

        fold_best_scores.append(search.best_score_)
        if return_estimators and fold_best_estimators is not None:
            fold_best_estimators.append(search.best_estimator_)
            fold_test_indices.append([int(i) for i in test_idx])

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

    fold_f1_external = [m["f1_macro"] for m in fold_metrics]
    result["best_scores_internal"] = fold_best_scores
    result["best_score_internal_mean"] = float(np.mean(fold_best_scores))
    result["best_score_internal_std"] = float(np.std(fold_best_scores, ddof=1)) if len(fold_best_scores) > 1 else 0.0
    result["fold_metrics"] = fold_metrics
    result["fold_f1_external"] = fold_f1_external
    result["fold_delta_sesgo"] = [i - e for i, e in zip(fold_best_scores, fold_f1_external)]
    result["delta_sesgo"] = float(np.mean(result["fold_delta_sesgo"]))

    result["stability_raw"] = float(max(0.0, min(1.0, 1.0 - result["f1_macro_std"])))
    result["icn"] = None
    result["icn_raw"] = None
    result["stability"] = result["stability_raw"]
    if return_estimators:
        if estimator_cache_dir is not None:
            from significance import save_estimators
            save_estimators(
                estimator_cache_dir,
                target=target_name,
                model_key=model_spec.key,
                payload={
                    "estimators": fold_best_estimators,
                    "test_indices": fold_test_indices,
                },
            )
        return result, fold_best_estimators
    return result


ICN_WEIGHTS = {
    "f1_macro": 0.40,
    "balanced_accuracy": 0.25,
    "recall_macro": 0.20,
    "precision_macro": 0.10,
    "stability": 0.05,
}


def _compute_icn_raw(item: dict[str, Any]) -> float:
    """ICN sin normalizar (fórmula del PDF del laboratorio).

    Los pesos del PDF son: 0.40 F1 + 0.25 BA + 0.20 Recall + 0.10 Precision
    + 0.05 Estabilidad. stability es 1 - std(F1 macro entre folds),
    recortado a [0, 1].

    El valor resultante esta en el mismo rango que las metricas originales
    (~[0, 1] en la practica), lo que permite comparar entre targets.
    """
    return float(
        ICN_WEIGHTS["f1_macro"] * item["f1_macro_mean"]
        + ICN_WEIGHTS["balanced_accuracy"] * item["balanced_accuracy_mean"]
        + ICN_WEIGHTS["recall_macro"] * item["recall_macro_mean"]
        + ICN_WEIGHTS["precision_macro"] * item["precision_macro_mean"]
        + ICN_WEIGHTS["stability"] * item["stability_raw"]
    )


def assign_icn(results: list[dict[str, Any]]) -> None:
    """Asigna dos ICN a cada modelo:

    - ``icn_raw``: formula cruda del PDF (sin normalizar). Comparable entre
      targets, pero dominado por la magnitud absoluta de F1 y BA en targets
      faciles.
    - ``icn``: cada componente normalizada min-max entre los modelos del
      mismo target, luego ponderada. Adecuado para ordenar modelos dentro
      de un mismo target, pero no comparable entre targets.
    - ``stability`` y ``stability_raw``: ``stability`` es la version
      normalizada (utilizada por ``icn``); ``stability_raw`` es 1 - std(F1)
      recortado a [0, 1].
    """
    implemented = [item for item in results if item["implemented"]]
    if not implemented:
        return

    for item in implemented:
        item["icn_raw"] = _compute_icn_raw(item)

    if len(implemented) == 1:
        item = implemented[0]
        item["icn"] = item["icn_raw"]
        item["icn_note"] = (
            "ICN directo porque solo hay un modelo implementado; coincide "
            "con icn_raw por falta de base de normalizacion."
        )
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
        item["icn_note"] = (
            "ICN normalizado min-max entre modelos del mismo target; util "
            "para ordenar modelos dentro del target, no entre targets."
        )


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


def compute_delta_sesgo(results: list[dict[str, Any]]) -> None:
    for item in results:
        if not item.get("implemented", False):
            continue
        internal = item.get("best_score_internal_mean")
        external = item.get("f1_macro_mean")
        if internal is not None and external is not None:
            item["delta_sesgo"] = internal - external
