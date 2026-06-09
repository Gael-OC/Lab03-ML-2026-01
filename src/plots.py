from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import TARGETS

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
MODEL_DISPLAY_NAMES = ["Regresión Logística", "SVM lineal", "SVM RBF", "Árbol de decisión", "K-NN"]


def generate_all_plots(results_by_target: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[plots] F1 macro heatmap...")
    plot_f1_macro_comparison(results_by_target, output_dir)

    print("[plots] Balanced accuracy heatmap...")
    plot_balanced_accuracy_comparison(results_by_target, output_dir)

    print("[plots] Matrices de confusion...")
    plot_confusion_matrices(results_by_target, output_dir)

    print("[plots] ICN comparacion...")
    plot_icn_comparison(results_by_target, output_dir)

    print("[plots] Metricas heatmap...")
    plot_metrics_heatmap(results_by_target, output_dir)

    print("[plots] Distribucion de clases...")
    plot_class_distribution_report(results_by_target, output_dir)

    print("[plots] Delta sesgo...")
    plot_delta_sesgo(results_by_target, output_dir)

    print("[plots] SVM kernel comparison...")
    plot_svm_kernel_comparison(results_by_target, output_dir)

    print("[plots] Dashboard resumen...")
    plot_summary_dashboard(results_by_target, output_dir)

    print(f"[plots] completo. Figuras en {output_dir}")


# Funciones auxiliares

def _extract_metric_matrix(
    results_by_target: dict[str, list[dict]],
    metric: str,
) -> pd.DataFrame:
    data: dict[str, list[float | None]] = {}
    for target, results in results_by_target.items():
        values: list[float | None] = []
        for item in results:
            values.append(item.get(metric))
        data[target] = values
    return pd.DataFrame(data, index=MODEL_DISPLAY_NAMES)


def _best_model_per_target(results_by_target: dict) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for target, results in results_by_target.items():
        implemented = [r for r in results if r["implemented"] and r.get("icn") is not None]
        if not implemented:
            continue
        best[target] = max(implemented, key=lambda r: r["icn"])
    return best


# Heatmap F1 macro

def plot_f1_macro_comparison(results_by_target: dict[str, list[dict]], output_dir: Path) -> None:
    matrix = _extract_metric_matrix(results_by_target, "f1_macro_mean")

    fig, ax = plt.subplots(figsize=(len(matrix.columns) * 1.2 + 1, len(matrix.index) * 0.8 + 1))
    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={"shrink": 0.6, "label": "F1 macro"},
        ax=ax,
    )
    ax.set_title("F1 macro por modelo y objetivo", fontweight="bold", fontsize=13)
    ax.set_ylabel("Modelo")
    ax.set_xlabel("Objetivo")
    fig.tight_layout()
    fig.savefig(output_dir / "f1_macro_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Heatmap balanced accuracy

def plot_balanced_accuracy_comparison(results_by_target: dict[str, list[dict]], output_dir: Path) -> None:
    matrix = _extract_metric_matrix(results_by_target, "balanced_accuracy_mean")

    fig, ax = plt.subplots(figsize=(len(matrix.columns) * 1.2 + 1, len(matrix.index) * 0.8 + 1))
    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={"shrink": 0.6, "label": "Balanced accuracy"},
        ax=ax,
    )
    ax.set_title("Balanced accuracy por modelo y objetivo", fontweight="bold", fontsize=13)
    ax.set_ylabel("Modelo")
    ax.set_xlabel("Objetivo")
    fig.tight_layout()
    fig.savefig(output_dir / "balanced_accuracy_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Matrices de confusión (mejor modelo por target)

def plot_confusion_matrices(results_by_target: dict[str, list[dict]], output_dir: Path) -> None:
    best_models = _best_model_per_target(results_by_target)
    n_targets = len(best_models)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    axes = axes.flatten()

    for idx, (target, best) in enumerate(best_models.items()):
        ax = axes[idx]
        cm = np.array(best["confusion_matrix"])
        labels = best["labels"]

        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
            linewidths=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={"shrink": 0.6, "label": "Recall (normalizado por fila)"},
            ax=ax,
        )
        ax.set_title(f"{target}\nMejor: {best['model_name']} (ICN={best['icn']:.3f})", fontweight="bold", fontsize=11)
        ax.set_ylabel("Real")
        ax.set_xlabel("Predicho")

    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Matrices de confusión del mejor modelo por objetivo", fontweight="bold", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Comparación ICN

def plot_icn_comparison(results_by_target: dict[str, list[dict]], output_dir: Path) -> None:
    data: list[dict[str, Any]] = []
    for target, results in results_by_target.items():
        for item in results:
            if not item["implemented"] or item.get("icn") is None:
                continue
            data.append({
                "target": target,
                "modelo": item["model_name"],
                "ICN": item["icn"],
            })

    df = pd.DataFrame(data)
    mean_icn = df["ICN"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="target", y="ICN", hue="modelo", ax=ax, palette="colorblind")
    ax.axhline(y=mean_icn, color="gray", linestyle="--", linewidth=1, label=f"Promedio: {mean_icn:.3f}")
    ax.set_title("Índice Comparativo Normalizado (ICN) por modelo y objetivo", fontweight="bold", fontsize=13)
    ax.set_xlabel("Objetivo")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "icn_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Heatmap de métricas por target

def plot_metrics_heatmap(results_by_target: dict[str, list[dict]], output_dir: Path) -> None:
    metric_keys = [
        ("f1_macro_mean", "F1"),
        ("balanced_accuracy_mean", "BA"),
        ("recall_macro_mean", "Recall"),
        ("precision_macro_mean", "Precision"),
        ("stability", "Estab."),
        ("icn", "ICN"),
    ]

    for target, results in results_by_target.items():
        data = {}
        for key, label in metric_keys:
            values = [item.get(key) for item in results if item["implemented"]]
            if values:
                data[label] = values

        if not data:
            continue

        matrix = pd.DataFrame(data, index=MODEL_DISPLAY_NAMES).astype(float)

        fig, ax = plt.subplots(figsize=(len(metric_keys) * 1.2 + 1, len(matrix.index) * 0.8 + 1))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            linewidths=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={"shrink": 0.6, "label": "Valor normalizado"},
            ax=ax,
        )
        ax.set_title(f"Métricas — {target}", fontweight="bold", fontsize=13)
        ax.set_ylabel("Modelo")
        fig.tight_layout()
        fig.savefig(output_dir / f"metrics_heatmap_{target}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# Distribución de clases (desde resultados)

def plot_class_distribution_report(results_by_target: dict[str, list[dict]], output_dir: Path) -> None:
    n_targets = len(results_by_target)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for idx, (target, results) in enumerate(results_by_target.items()):
        ax = axes[idx]
        distribution = results[0].get("class_distribution", {})
        labels = list(distribution.keys())
        counts = list(distribution.values())
        n_min = results[0].get("n_min", min(counts))
        k_outer = results[0].get("k_outer", 0)
        total = sum(counts)

        bars = ax.bar(
            [str(int(l)) for l in labels],
            counts,
            color=sns.color_palette("colorblind", len(labels)),
            edgecolor="white",
            linewidth=0.8,
        )

        for bar, count in zip(bars, counts):
            pct = 100 * count / total
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_title(f"{target}\nn_min={n_min}, k_outer={k_outer}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Clase")
        ax.set_ylabel("Frecuencia")

    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Distribución de clases por variable objetivo", fontweight="bold", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution_report.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Delta sesgo

def plot_delta_sesgo(results_by_target: dict[str, list[dict]], output_dir: Path) -> None:
    data: list[dict[str, Any]] = []
    for target, results in results_by_target.items():
        for item in results:
            ds = item.get("delta_sesgo")
            if not item["implemented"] or ds is None:
                continue
            data.append({
                "target": target,
                "modelo": item["model_name"],
                "delta_sesgo": ds,
            })

    if not data:
        return

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="target", y="delta_sesgo", hue="modelo", ax=ax, palette="colorblind")
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_title("Δsesgo (score interno − score externo) por modelo y objetivo", fontweight="bold", fontsize=13)
    ax.set_xlabel("Objetivo")
    ax.set_ylabel("Δsesgo")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "delta_sesgo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Comparación kernel SVM lineal vs RBF

def plot_svm_kernel_comparison(results_by_target: dict[str, list[dict]], output_dir: Path) -> None:
    targets = sorted(results_by_target.keys())
    linear_f1: list[float] = []
    rbf_f1: list[float] = []

    for target in targets:
        results = results_by_target[target]
        linear = next((r for r in results if r["model_key"] == "svm_linear" and r["implemented"]), None)
        rbf = next((r for r in results if r["model_key"] == "svm_rbf" and r["implemented"]), None)
        if linear and rbf:
            linear_f1.append(linear["f1_macro_mean"])
            rbf_f1.append(rbf["f1_macro_mean"])

    if not linear_f1:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(linear_f1, rbf_f1, s=80, color=sns.color_palette("colorblind")[2])
    for i, target in enumerate(targets):
        if i < len(linear_f1):
            ax.annotate(target, (linear_f1[i], rbf_f1[i]), fontsize=9, ha="center", va="bottom")

    min_val = min(min(linear_f1), min(rbf_f1)) - 0.02
    max_val = max(max(linear_f1), max(rbf_f1)) + 0.02
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.5, label="y = x")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("F1 macro — SVM lineal", fontsize=12)
    ax.set_ylabel("F1 macro — SVM RBF", fontsize=12)
    ax.set_title("Comparación SVM lineal vs SVM RBF\n(puntos sobre la línea → RBF gana)", fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "svm_kernel_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# Dashboard resumen

def plot_summary_dashboard(results_by_target: dict[str, list[dict]], output_dir: Path) -> None:
    metrics = ["f1_macro_mean", "balanced_accuracy_mean", "recall_macro_mean", "precision_macro_mean", "stability"]
    metric_labels = ["F1 macro", "BalAcc", "Recall", "Precision", "Estabilidad"]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    n_targets = len(results_by_target)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), subplot_kw={"projection": "polar"})
    axes = axes.flatten()

    for idx, (target, results) in enumerate(results_by_target.items()):
        ax = axes[idx]
        implemented = [r for r in results if r["implemented"]]

        for item in implemented:
            values = [item.get(m, 0) or 0 for m in metrics]
            values += values[:1]
            ax.plot(angles, values, marker="o", label=item["model_name"], linewidth=1.5)
            ax.fill(angles, values, alpha=0.05)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=7)
        ax.set_title(target, fontsize=13, fontweight="bold", pad=15)
        ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Dashboard resumen — métricas por objetivo", fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
