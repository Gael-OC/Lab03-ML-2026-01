from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import FEATURE_COLS, TARGETS

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(df, TARGETS, output_dir)
    plot_correlation_heatmap(df, FEATURE_COLS, output_dir)
    plot_target_relationship(df, TARGETS, output_dir)
    generate_eda_summary_table(df, TARGETS, FEATURE_COLS, output_dir)

    print(f"EDA completado. Figuras guardadas en {output_dir}")


def plot_class_distribution(df: pd.DataFrame, targets: list[str], output_dir: Path) -> None:
    n_targets = len(targets)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for idx, target in enumerate(targets):
        ax = axes[idx]
        counts = df[target].value_counts().sort_index()
        n_min = int(counts.min())
        k_outer = min(5, n_min)
        total = int(counts.sum())

        bars = ax.bar(
            [str(int(c)) for c in counts.index],
            counts.values,
            color=sns.color_palette("colorblind", len(counts)),
            edgecolor="white",
            linewidth=0.8,
        )

        for bar, count in zip(bars, counts.values):
            pct = 100 * count / total
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts.values) * 0.02,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(f"{target}\nn_min={n_min}, k_outer={k_outer}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Clase")
        ax.set_ylabel("Frecuencia")
        ax.set_ylim(0, max(counts.values) * 1.25)

    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Distribución de clases por variable objetivo", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list[str], output_dir: Path) -> None:
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Coeficiente de Pearson (phi para binarias)"},
        ax=ax,
    )

    ax.set_title(
        "Correlación entre atributos binarios\n(Pearson = phi coefficient para variables dicotómicas)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "feature_correlation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_target_relationship(df: pd.DataFrame, targets: list[str], output_dir: Path) -> None:
    gds_targets = [t for t in targets if t != "GDS"]
    n_plots = len(gds_targets)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for idx, target in enumerate(gds_targets):
        ax = axes[idx]
        ctab = pd.crosstab(df["GDS"], df[target])

        sns.heatmap(
            ctab,
            annot=True,
            fmt="d",
            cmap="Blues",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Conteo"},
            ax=ax,
        )

        ax.set_title(f"GDS → {target}", fontsize=13, fontweight="bold")
        ax.set_xlabel(target)
        ax.set_ylabel("GDS (original)")

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Relación entre GDS original y sus agrupaciones derivadas",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "target_relationships.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_eda_summary_table(
    df: pd.DataFrame,
    targets: list[str],
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    rows = []

    for target in targets:
        counts = df[target].value_counts().sort_index()
        n_min = int(counts.min())
        k_outer = min(5, n_min)
        n_clases = len(counts)
        ratio = float(counts.max() / counts.min()) if n_clases > 1 else 1.0
        distribucion = ", ".join(f"{int(k)}:{int(v)}" for k, v in counts.items())

        rows.append({
            "target": target,
            "n_clases": n_clases,
            "n_min": n_min,
            "k_outer": k_outer,
            "ratio_desbalance": f"{ratio:.1f}",
            "distribucion": distribucion,
        })

    summary_df = pd.DataFrame(rows)

    general_stats = {
        "target": "[GENERAL]",
        "n_clases": "",
        "n_min": "",
        "k_outer": "",
        "ratio_desbalance": "",
        "distribucion": (
            f"n_muestras={len(df)}, n_features={len(feature_cols)}, "
            f"missing={int(df[feature_cols].isnull().sum().sum())}, "
            f"duplicated={int(df.duplicated().sum())}"
        ),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([general_stats])], ignore_index=True)

    summary_df.to_csv(output_dir / "eda_summary.csv", index=False, encoding="utf-8")
