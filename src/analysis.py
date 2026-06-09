from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import KFold, learning_curve
from sklearn.tree import plot_tree

from data_loader import FEATURE_COLS, TARGETS
from models import build_model_registry

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.0)
MODEL_ORDER = ["logistic", "svm_linear", "svm_rbf", "tree", "knn"]


def run_advanced_analysis(
    results_by_target: dict[str, list[dict[str, Any]]],
    df: pd.DataFrame,
    output_dirs: dict[str, Path],
) -> None:
    figures_dir = output_dirs.get("analysis", output_dirs["figures"] / "analysis")
    tables_dir = output_dirs["tables"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[analysis] learning curves...")
    analyze_learning_curves(df, figures_dir)

    print("[analysis] feature importance...")
    analyze_feature_importance(df, figures_dir)

    print("[analysis] hyperparameter stability...")
    analyze_hyperparameter_stability(results_by_target, figures_dir)

    print("[analysis] zero-recall classes...")
    detect_zero_recall_classes(results_by_target, tables_dir, figures_dir)

    print("[analysis] low-support classes...")
    detect_low_support_classes(results_by_target, tables_dir)

    print("[analysis] decision tree visualization...")
    visualize_decision_trees(df, figures_dir)

    print("[analysis] per-fold stability...")
    analyze_per_fold_stability(results_by_target, figures_dir)

    print(f"[analysis] completo. Figuras en {figures_dir}")


# Curvas de aprendizaje

def analyze_learning_curves(df: pd.DataFrame, output_dir: Path) -> None:
    registry = build_model_registry(random_state=42)
    n_targets = len(TARGETS)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        X = df[FEATURE_COLS].astype(float)
        y = df[target].astype(int)

        counts = y.value_counts()
        n_min = int(counts.min())
        k_cv = max(2, min(5, n_min))

        for model_key in MODEL_ORDER:
            spec = registry[model_key]
            if not spec.implemented:
                continue

            try:
                train_sizes, train_scores, test_scores = learning_curve(
                    clone(spec.pipeline),
                    X, y,
                    train_sizes=np.linspace(0.1, 1.0, 8),
                    cv=KFold(n_splits=k_cv, shuffle=True, random_state=42),
                    scoring="f1_macro",
                    n_jobs=-1,
                    error_score=0.0,
                )
                mean_test = test_scores.mean(axis=1)
                std_test = test_scores.std(axis=1)
                ax.plot(train_sizes, mean_test, marker="o", label=spec.display_name, linewidth=1.5)
                ax.fill_between(train_sizes, mean_test - std_test, mean_test + std_test, alpha=0.1)
            except Exception:
                continue

        ax.set_title(f"{target} (k={k_cv})", fontweight="bold")
        ax.set_xlabel("Tamaño del entrenamiento")
        ax.set_ylabel("F1 macro")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    for idx in range(len(TARGETS), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Curvas de aprendizaje por target y modelo", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Importancia de atributos

def analyze_feature_importance(df: pd.DataFrame, output_dir: Path) -> None:
    registry = build_model_registry(random_state=42)
    importance_models = [
        ("logistic", "Regresión Logística"),
        ("svm_linear", "SVM lineal"),
        ("tree", "Árbol de decisión"),
    ]

    for target in TARGETS:
        X = df[FEATURE_COLS].astype(float)
        y = df[target].astype(int)

        importance_data: dict[str, np.ndarray] = {}

        for model_key, display_name in importance_models:
            spec = registry[model_key]
            try:
                model = clone(spec.pipeline)
                model.fit(X, y)
                clf = model.named_steps["clf"]

                if model_key == "tree":
                    imp = clf.feature_importances_
                else:
                    imp = np.abs(clf.coef_).mean(axis=0) if clf.coef_.ndim > 1 else np.abs(clf.coef_)

                imp = imp / (imp.sum() + 1e-12)
                importance_data[display_name] = imp
            except Exception:
                importance_data[display_name] = np.zeros(len(FEATURE_COLS))

        imp_df = pd.DataFrame(importance_data, index=FEATURE_COLS)

        n_models = len(importance_models)
        n_features = len(FEATURE_COLS)
        fig, ax = plt.subplots(
            figsize=(n_models * 2.5 + 2, n_features * 0.45 + 2),
            constrained_layout=True,
        )
        sns.heatmap(
            imp_df,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            linewidths=0.5,
            cbar_kws={"shrink": 0.6, "label": "Importancia normalizada"},
            ax=ax,
        )
        ax.set_title(f"Importancia de atributos — {target}", fontweight="bold", fontsize=13)
        ax.set_ylabel("Atributo")
        fig.savefig(output_dir / f"feature_importance_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# Estabilidad de hiperparámetros

def analyze_hyperparameter_stability(results_by_target: dict, output_dir: Path) -> None:
    for target, results in results_by_target.items():
        data: list[dict[str, Any]] = []
        for item in results:
            if not item["implemented"]:
                continue
            params_str = item.get("best_params_mode", "")
            data.append({
                "modelo": item["model_name"],
                "hiperparametros": params_str,
            })

        if not data:
            continue

        hp_df = pd.DataFrame(data)
        pivot = hp_df.groupby(["modelo", "hiperparametros"]).size().unstack(fill_value=0)

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot.index) * 0.8)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt="d",
            cmap="Blues",
            linewidths=0.5,
            cbar_kws={"shrink": 0.6, "label": "Frecuencia (sobre folds externos)"},
            ax=ax,
        )
        ax.set_title(f"Estabilidad de hiperparámetros — {target}", fontweight="bold", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / f"hyperparam_stability_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# Clases con recall cero

def detect_zero_recall_classes(
    results_by_target: dict,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for target, results in results_by_target.items():
        for item in results:
            if not item["implemented"] or not item.get("classification_report"):
                continue
            report = item["classification_report"]
            for label in item["labels"]:
                class_metrics = report.get(str(label), {})
                recall = class_metrics.get("recall", 0)
                if recall == 0.0:
                    rows.append({
                        "target": target,
                        "modelo": item["model_name"],
                        "clase": label,
                        "precision": class_metrics.get("precision", 0),
                        "recall": 0.0,
                        "f1_score": class_metrics.get("f1-score", 0),
                        "support": int(class_metrics.get("support", 0)),
                    })

    if not rows:
        pd.DataFrame(columns=["target", "modelo", "clase", "precision", "recall", "f1_score", "support"]) \
            .to_csv(tables_dir / "zero_recall_classes.csv", index=False, encoding="utf-8")
        return

    zero_df = pd.DataFrame(rows)
    zero_df.to_csv(tables_dir / "zero_recall_classes.csv", index=False, encoding="utf-8")

    pivot = zero_df.pivot_table(
        index=["target", "clase"],
        columns="modelo",
        values="recall",
        aggfunc="first",
    )
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(4, len(pivot.index) * 0.5)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        linewidths=0.5,
        cbar_kws={"shrink": 0.6, "label": "Recall (0 = clase nunca predicha)"},
        ax=ax,
    )
    ax.set_title("Clases con recall = 0 por target y modelo", fontweight="bold", fontsize=12)
    ax.set_ylabel("Target · Clase")
    fig.tight_layout()
    fig.savefig(figures_dir / "zero_recall_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Clases con soporte bajo

def detect_low_support_classes(
    results_by_target: dict,
    tables_dir: Path,
    min_support: int = 10,
) -> None:
    rows: list[dict[str, Any]] = []
    for target, results in results_by_target.items():
        distribution = results[0].get("class_distribution", {})
        for label, support in distribution.items():
            if support >= min_support:
                continue
            for item in results:
                recall = None
                precision = None
                f1 = None
                zero_recall = False
                if item["implemented"] and item.get("classification_report"):
                    cr = item["classification_report"]
                    class_m = cr.get(str(int(label)), {})
                    recall = class_m.get("recall")
                    precision = class_m.get("precision")
                    f1 = class_m.get("f1-score")
                    zero_recall = (recall == 0.0)
                rows.append({
                    "target": target,
                    "clase": int(label),
                    "support": int(support),
                    "modelo": item["model_name"],
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "zero_recall": zero_recall,
                })

    columns = ["target", "clase", "support", "modelo", "precision", "recall", "f1_score", "zero_recall"]
    ls_df = pd.DataFrame(rows, columns=columns)
    ls_df.to_csv(tables_dir / "low_support_classes.csv", index=False, encoding="utf-8")


# Visualización del árbol de decisión

def visualize_decision_trees(df: pd.DataFrame, output_dir: Path) -> None:
    from sklearn.tree import DecisionTreeClassifier

    for target in TARGETS:
        X = df[FEATURE_COLS].astype(float)
        y = df[target].astype(int)
        unique_classes = sorted(y.unique())

        tree = DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
            max_depth=4,
        )
        tree.fit(X, y)

        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(
            tree,
            feature_names=FEATURE_COLS,
            class_names=[str(c) for c in unique_classes],
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax,
        )
        ax.set_title(f"Árbol de decisión — {target} (max_depth=4)", fontweight="bold", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f"decision_tree_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# Estabilidad por fold

def analyze_per_fold_stability(results_by_target: dict, output_dir: Path) -> None:
    n_targets = len(TARGETS)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    for idx, (target, results) in enumerate(results_by_target.items()):
        ax = axes[idx]
        data: list[dict[str, Any]] = []

        for item in results:
            if not item["implemented"] or not item.get("fold_metrics"):
                continue
            for fold_idx, fm in enumerate(item["fold_metrics"]):
                data.append({
                    "modelo": item["model_name"],
                    "fold": fold_idx + 1,
                    "f1_macro": fm["f1_macro"],
                })

        if not data:
            ax.set_title(target)
            ax.text(0.5, 0.5, "Sin datos por fold", ha="center", va="center", transform=ax.transAxes)
            continue

        df_plot = pd.DataFrame(data)
        sns.boxplot(data=df_plot, x="modelo", y="f1_macro", hue="modelo", ax=ax, palette="colorblind", legend=False)
        ax.set_title(target, fontweight="bold")
        ax.set_ylabel("F1 macro")
        ax.tick_params(axis="x", rotation=15)

    for idx in range(len(TARGETS), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Distribución de F1 macro por fold externo", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "per_fold_f1_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
