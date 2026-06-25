from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve
from sklearn.tree import plot_tree

from data_loader import FEATURE_COLS, TARGETS
from models import MODEL_ORDER, build_model_registry
from significance import load_estimators, run_significance_tests

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.0)
MODEL_DISPLAY_NAMES = {
    "dummy": "Baseline (Dummy)",
    "logistic": "Regresion Logistica",
    "svm_linear": "SVM lineal",
    "svm_rbf": "SVM RBF",
    "tree": "Arbol de decision",
    "knn": "K-NN",
}


def run_advanced_analysis(
    results_by_target: dict[str, list[dict[str, Any]]],
    df: pd.DataFrame,
    output_dirs: dict[str, Path],
    estimator_cache_dir: Path | None = None,
) -> None:
    figures_dir = output_dirs.get("analysis", output_dirs["figures"] / "analysis")
    tables_dir = output_dirs["tables"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[analysis] learning curves...")
    analyze_learning_curves(df, figures_dir)

    print("[analysis] feature importance...")
    analyze_feature_importance(df, figures_dir, estimator_cache_dir=estimator_cache_dir)

    print("[analysis] hyperparameter stability...")
    analyze_hyperparameter_stability(results_by_target, figures_dir)

    print("[analysis] zero-recall classes...")
    detect_zero_recall_classes(results_by_target, tables_dir, figures_dir)

    print("[analysis] low-support classes...")
    detect_low_support_classes(results_by_target, tables_dir)

    print("[analysis] decision tree visualization...")
    visualize_decision_trees(df, figures_dir, estimator_cache_dir=estimator_cache_dir)

    print("[analysis] per-fold stability...")
    analyze_per_fold_stability(results_by_target, figures_dir)

    print("[analysis] bootstrap CI...")
    analyze_bootstrap_ci(results_by_target, tables_dir, figures_dir)

    if estimator_cache_dir is not None:
        print("[analysis] statistical tests (Wilcoxon + McNemar-Yates)...")
        analyze_statistical_tests(
            results_by_target, df, tables_dir, figures_dir, estimator_cache_dir
        )
    else:
        print("[analysis] tests de significancia omitidos (sin cache de estimadores).")
        print("          Ejecute `python src/main.py --keep-estimators` para habilitarlos.")

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
                    cv=StratifiedKFold(n_splits=k_cv, shuffle=True, random_state=42),
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

def _extract_importance(estimator, model_key: str) -> np.ndarray:
    """Extrae la importancia de un estimador ya entrenado.

    Para modelos lineales (LR, SVM lineal) se usa el valor absoluto de
    los coeficientes promediado sobre las clases. Para Arbol se usa
    ``feature_importances_`` de sklearn.
    """
    clf = estimator.named_steps["clf"]
    if model_key == "tree":
        return clf.feature_importances_
    if clf.coef_.ndim > 1:
        return np.abs(clf.coef_).mean(axis=0)
    return np.abs(clf.coef_).flatten()


def analyze_feature_importance(
    df: pd.DataFrame,
    output_dir: Path,
    estimator_cache_dir: Path | None = None,
) -> None:
    """Heatmap de importancia por feature y por modelo.

    Si hay cache de estimadores (modo ``--keep-estimators``), la
    importancia de cada feature se promedia sobre los ``best_estimator_``
    de los folds externos, lo que da una estimacion mas estable que
    ajustar un solo modelo sobre todo el dataset. Si no hay cache, se
    cae al ajuste global como fallback (mas rapido pero menos estable).
    """
    registry = build_model_registry(random_state=42)
    importance_models = [
        ("logistic", "Regresion Logistica"),
        ("svm_linear", "SVM lineal"),
        ("tree", "Arbol de decision"),
    ]

    for target in TARGETS:
        importance_data: dict[str, np.ndarray] = {}

        for model_key, display_name in importance_models:
            per_fold_imps: list[np.ndarray] = []
            if estimator_cache_dir is not None:
                payload = load_estimators(estimator_cache_dir, target, model_key)
                if payload is not None:
                    for est in payload["estimators"]:
                        try:
                            per_fold_imps.append(_extract_importance(est, model_key))
                        except Exception:
                            continue

            if per_fold_imps:
                imp = np.mean(per_fold_imps, axis=0)
            else:
                # Fallback: ajustar un modelo global.
                spec = registry[model_key]
                try:
                    model = clone(spec.pipeline)
                    model.fit(df[FEATURE_COLS].astype(float), df[target].astype(int))
                    imp = _extract_importance(model, model_key)
                except Exception:
                    imp = np.zeros(len(FEATURE_COLS))

            imp = imp / (imp.sum() + 1e-12)
            importance_data[display_name] = imp

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
        ax.set_title(f"Importancia de atributos (promedio sobre folds) - {target}", fontweight="bold", fontsize=13)
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

def visualize_decision_trees(
    df: pd.DataFrame,
    output_dir: Path,
    estimator_cache_dir: Path | None = None,
) -> None:
    """Visualiza el árbol de decisión por target.

    Si hay cache de estimadores, usa el best_estimator_ del primer
    fold externo (que ya tiene los hiperparametros seleccionados por
    GridSearchCV y el scaler entrenado en su fold de entrenamiento).
    Si no hay cache, entrena un Arbol global con max_depth=4 como
    fallback. La visualizacion usa el Arbol del Pipeline (sin scaler,
    porque el Arbol no lo usa).
    """
    for target in TARGETS:
        X = df[FEATURE_COLS].astype(float)
        y = df[target].astype(int)
        unique_classes = sorted(y.unique())

        tree = None
        if estimator_cache_dir is not None:
            payload = load_estimators(estimator_cache_dir, target, "tree")
            if payload is not None and payload["estimators"]:
                tree = payload["estimators"][0].named_steps["clf"]

        if tree is None:
            from sklearn.tree import DecisionTreeClassifier
            tree = DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42,
                max_depth=4,
            )
            tree.fit(X, y)
            title_suffix = "max_depth=4 (fallback)"
        else:
            depth = tree.get_depth()
            title_suffix = f"mejor fold externo (max_depth={depth})"

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
        ax.set_title(f"Árbol de decisión — {target} ({title_suffix})", fontweight="bold", fontsize=14)
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


# Bootstrap: intervalos de confianza al 95% por remuestreo

def bootstrap_ci(metric_values: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Calcula el intervalo de confianza por el método bootstrap de percentiles.

    Argumentos:
        metric_values: lista de valores de la métrica por fold externo.
        n_bootstrap: cantidad de remuestras con reemplazo.
        ci: nivel de confianza (0.95 por defecto).

    Retorna:
        (limite_inferior, limite_superior) del IC.
    """
    arr = np.asarray(metric_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed=42)
    boot_means = np.empty(n_bootstrap)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lower, upper)


def analyze_bootstrap_ci(
    results_by_target: dict[str, list[dict[str, Any]]],
    tables_dir: Path,
    figures_dir: Path,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> None:
    """Calcula IC bootstrap al 95% para F1, balanced accuracy y recall por modelo y target.

    Genera una tabla CSV y un forest plot comparativo. La metodología sigue el bootstrap
    de percentiles descrito en Clase 12.
    """
    metrics = [
        ("f1_macro", "F1 macro"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("recall_macro", "Recall macro"),
    ]

    rows: list[dict[str, Any]] = []
    forest_data: list[dict[str, Any]] = []

    for target, results in results_by_target.items():
        for item in results:
            if not item["implemented"] or not item.get("fold_metrics"):
                continue
            for metric_key, metric_label in metrics:
                values = [m.get(metric_key) for m in item["fold_metrics"]]
                values = [v for v in values if v is not None]
                if not values:
                    continue
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ci_low, ci_high = bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=ci)
                rows.append({
                    "target": target,
                    "modelo": item["model_name"],
                    "metrica": metric_label,
                    "mean": mean_val,
                    "std": std_val,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "n_folds": len(values),
                })
                if metric_key == "f1_macro":
                    forest_data.append({
                        "target": target,
                        "modelo": item["model_name"],
                        "mean": mean_val,
                        "ci_lower": ci_low,
                        "ci_upper": ci_high,
                    })

    ci_pct = int(ci * 100)
    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(tables_dir / f"bootstrap_ci_{ci_pct}.csv", index=False, encoding="utf-8")

    if not forest_data:
        return

    forest_df = pd.DataFrame(forest_data)
    n_targets = len(TARGETS)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 4.5))
    axes = axes.flatten()
    palette = sns.color_palette("colorblind", len(MODEL_ORDER))
    color_map = {MODEL_DISPLAY_NAMES[key]: palette[i] for i, key in enumerate(MODEL_ORDER)}

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        sub = forest_df[forest_df["target"] == target].copy()
        if sub.empty:
            ax.set_title(target)
            continue

        order = sub.sort_values("mean", ascending=True)
        y_pos = np.arange(len(order))
        for y_i, (_, row) in enumerate(order.iterrows()):
            color = color_map.get(row["modelo"], palette[0])
            ax.errorbar(
                row["mean"],
                y_i,
                xerr=[[row["mean"] - row["ci_lower"]], [row["ci_upper"] - row["mean"]]],
                fmt="o",
                color=color,
                capsize=4,
                markersize=6,
                linewidth=1.5,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(order["modelo"].tolist(), fontsize=8)
        ax.set_title(target, fontweight="bold")
        ax.set_xlabel("F1 macro")
        ax.grid(True, axis="x", alpha=0.3)

    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"F1 macro con IC bootstrap al {ci_pct}% por modelo y objetivo",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / f"bootstrap_ci_forest_{ci_pct}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# 4.10  Tests de significancia estadistica: Wilcoxon + McNemar (Yates)

def analyze_statistical_tests(
    results_by_target: dict[str, list[dict[str, Any]]],
    df: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    estimator_cache_dir: Path,
    alpha: float = 0.05,
) -> None:
    """Wilcoxon firmado sobre F1 por fold + McNemar con correccion de Yates.

    Usa los estimadores persistidos en `estimator_cache_dir` por `run_nested_cv`,
    por lo que no reentrena la validacion anidada. Esto desacopla el costo del
    analisis estadistico del costo de los experimentos.

    Wilcoxon: test no parametrico pareado, valido con muestras pequenas.
    McNemar-Yates: discrepacias de acierto/fallo entre dos modelos en los
    mismos ejemplos, con correccion de continuidad recomendada cuando
    b + c < 25 (comun con k=5 folds).
    """
    target_names = [t for t in results_by_target if t in TARGETS]

    rows_df = run_significance_tests(
        df=df,
        results_by_target=results_by_target,
        cache_dir=estimator_cache_dir,
        targets=target_names,
        model_order=MODEL_ORDER,
        model_display_names=MODEL_DISPLAY_NAMES,
        alpha=alpha,
    )

    if not rows_df.empty:
        rows_df.to_csv(
            tables_dir / "significance_tests.csv", index=False, encoding="utf-8"
        )

    heatmap_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for target in target_names:
        sub = rows_df[rows_df["target"] == target]
        if sub.empty:
            continue
        models = sorted(set(sub["modelo_A"]).union(sub["modelo_B"]))
        p_w = pd.DataFrame(np.nan, index=models, columns=models)
        p_m = pd.DataFrame(np.nan, index=models, columns=models)
        for _, row in sub.iterrows():
            p_w.loc[row["modelo_A"], row["modelo_B"]] = row["p_wilcoxon"]
            p_w.loc[row["modelo_B"], row["modelo_A"]] = row["p_wilcoxon"]
            p_m.loc[row["modelo_A"], row["modelo_B"]] = row["p_mcnemar_yates"]
            p_m.loc[row["modelo_B"], row["modelo_A"]] = row["p_mcnemar_yates"]
        for name in models:
            p_w.loc[name, name] = np.nan
            p_m.loc[name, name] = np.nan
        heatmap_data[target] = (p_w, p_m)

    _plot_significance_heatmaps(heatmap_data, MODEL_DISPLAY_NAMES, figures_dir, alpha)


def _plot_significance_heatmaps(
    heatmap_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    model_names: dict[str, str],
    figures_dir: Path,
    alpha: float,
) -> None:
    n_targets = len(heatmap_data)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    axes = axes.flatten()

    for idx, (target, (p_w, p_m)) in enumerate(heatmap_data.items()):
        ax = axes[idx]
        if p_w.empty:
            ax.set_title(target)
            continue

        display = p_w.rename(index=model_names, columns=model_names)
        annot = display.copy().astype(object)
        for r in range(display.shape[0]):
            for c in range(display.shape[1]):
                val = display.iloc[r, c]
                if np.isnan(val):
                    annot.iloc[r, c] = ""
                elif r == c:
                    annot.iloc[r, c] = "—"
                else:
                    sig = "*" if val < alpha else ""
                    annot.iloc[r, c] = f"{val:.3f}{sig}"

        sns.heatmap(
            display.astype(float),
            annot=annot,
            fmt="",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            cbar_kws={"shrink": 0.6, "label": f"p-value (Wilcoxon)"},
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f"{target}\nWilcoxon (* = p<{alpha})", fontweight="bold", fontsize=11)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Wilcoxon firmado: p-values por par de modelos y objetivo",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "significance_heatmap_wilcoxon.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    axes = axes.flatten()

    for idx, (target, (p_w, p_m)) in enumerate(heatmap_data.items()):
        ax = axes[idx]
        if p_m.empty:
            ax.set_title(target)
            continue

        display = p_m.rename(index=model_names, columns=model_names)
        annot = display.copy().astype(object)
        for r in range(display.shape[0]):
            for c in range(display.shape[1]):
                val = display.iloc[r, c]
                if np.isnan(val):
                    annot.iloc[r, c] = ""
                elif r == c:
                    annot.iloc[r, c] = "—"
                else:
                    sig = "*" if val < alpha else ""
                    annot.iloc[r, c] = f"{val:.3f}{sig}"

        sns.heatmap(
            display.astype(float),
            annot=annot,
            fmt="",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            cbar_kws={"shrink": 0.6, "label": f"p-value (McNemar-Yates)"},
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f"{target}\nMcNemar-Yates (* = p<{alpha})", fontweight="bold", fontsize=11)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "McNemar con correccion de Yates: p-values por par de modelos y objetivo",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "significance_heatmap_mcnemar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
