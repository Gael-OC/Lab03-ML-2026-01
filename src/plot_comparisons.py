"""Genera graficos comparativos de las mejoras."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import TARGETS

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.0)

EXPERIMENTS = ["BASE", "FE", "GROUP", "FE+GROUP"]
EXP_COLORS = {
    "BASE": "#888888",
    "FE": "#117733",
    "GROUP": "#882255",
    "FE+GROUP": "#DDCC77",
}


def main() -> None:
    df = pd.read_csv("outputs/comparisons/comparison_baseline_vs_improvements.csv")

    targets_to_plot = []
    for target in TARGETS:
        sub = df[df["target"] == target]
        for exp in EXPERIMENTS:
            if not sub[sub["experimento"] == exp].empty:
                targets_to_plot.append(target)
                break

    n = len(targets_to_plot)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    axes = axes.flatten()

    for idx, target in enumerate(targets_to_plot):
        ax = axes[idx]
        sub = df[df["target"] == target]
        models = sub[sub["experimento"] == "BASE"]["modelo"].tolist()

        x = np.arange(len(models))
        width = 0.2
        for i, exp in enumerate(EXPERIMENTS):
            sub_exp = sub[sub["experimento"] == exp]
            f1s = [sub_exp[sub_exp["modelo"] == m]["f1_macro_mean"].iloc[0]
                   if not sub_exp[sub_exp["modelo"] == m].empty else 0
                   for m in models]
            ax.bar(x + (i - 1.5) * width, f1s, width, label=exp, color=EXP_COLORS[exp])

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_title(target, fontweight="bold")
        ax.set_ylabel("F1 macro")
        ax.set_ylim(0, max(0.85, sub["f1_macro_mean"].max() * 1.15))

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.suptitle(
        "Comparacion BASE vs FE vs GROUP vs FE+GROUP: F1 macro por modelo",
        fontsize=15, fontweight="bold", y=1.08,
    )
    fig.tight_layout()

    out_dir = Path("outputs/figures/comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "f1_comparison_baseline_vs_improvements.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico guardado en {out_dir / 'f1_comparison_baseline_vs_improvements.png'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_data = []
    for target in targets_to_plot:
        for exp in EXPERIMENTS:
            sub = df[(df["target"] == target) & (df["experimento"] == exp)]
            if sub.empty:
                continue
            best = sub.loc[sub["f1_macro_mean"].idxmax()]
            bar_data.append({
                "target": target,
                "experimento": exp,
                "f1_macro": best["f1_macro_mean"],
                "modelo": best["modelo"],
            })
    bar_df = pd.DataFrame(bar_data)
    pivot = bar_df.pivot(index="target", columns="experimento", values="f1_macro")
    pivot = pivot[EXPERIMENTS]
    pivot.plot(kind="bar", ax=ax, color=[EXP_COLORS[e] for e in EXPERIMENTS])
    ax.set_ylabel("Mejor F1 macro")
    ax.set_title("Mejor F1 macro por target y experimento", fontweight="bold")
    ax.legend(title="Experimento")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / "best_f1_per_target.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico guardado en {out_dir / 'best_f1_per_target.png'}")


if __name__ == "__main__":
    main()
