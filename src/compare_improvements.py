"""Compara resultados con/sin feature engineering y con/sin agrupacion de clases raras.

Genera una tabla comparativa para cada target con ambos enfoques.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    FEATURE_COLS,
    TARGETS,
    group_rare_gds_classes,
    load_sav_dataset,
    prepare_xy,
)
from evaluation import assign_icn, compute_delta_sesgo, compute_outer_folds, run_nested_cv, unimplemented_result
from models import MODEL_ORDER, build_model_registry
from settings import DEFAULT_CONFIG_PATH, load_config


CONFIG = {
    "max_outer_folds": 5,
    "max_inner_folds": 3,
    "outer_random_state": 42,
    "inner_random_state": 123,
    "n_jobs": -1,
}


def comparisons_dir() -> Path:
    """Ruta al directorio de tablas y figuras del analisis de mejoras."""
    config = load_config(DEFAULT_CONFIG_PATH)
    return Path(config["outputs"]["root"]) / "comparisons"

N_INTERACTIONS = 21  # Combinaciones de a pares dentro de los 4 grupos semanticos


def run_all(df: pd.DataFrame, use_interactions: bool, group_gds: bool, label: str) -> dict:
    df_work = group_rare_gds_classes(df, threshold=50) if group_gds else df.copy()
    registry = build_model_registry(random_state=42)
    results = {}

    for target in TARGETS:
        if group_gds and target != "GDS":
            continue

        X, y = prepare_xy(df_work, target, use_interactions=use_interactions)
        n_features = X.shape[1]
        n_min, k_outer = compute_outer_folds(y, CONFIG["max_outer_folds"])
        distribution = {int(k): int(v) for k, v in y.value_counts().sort_index().items()}
        target_results = []

        for model_key in MODEL_ORDER:
            spec = registry[model_key]
            if spec.implemented:
                result = run_nested_cv(X, y, target, spec, CONFIG)
            else:
                result = unimplemented_result(target, spec, distribution, n_min, k_outer)
            result["n_features"] = n_features
            target_results.append(result)

        assign_icn(target_results)
        compute_delta_sesgo(target_results)
        results[target] = target_results
        print(f"  {label} | {target} | n_min={n_min} | n_features={n_features} | F1_max={max(r['f1_macro_mean'] or 0 for r in target_results):.4f}")

    return results


def main() -> None:
    df = load_sav_dataset("datasets/15 atributos R0-R5.sav")

    print("\n[BASELINE] Sin mejoras, todos los targets")
    base = run_all(df, use_interactions=False, group_gds=False, label="BASE")

    print("\n[FE] Con 21 interacciones AND dentro de cada grupo semantico")
    with_fe = run_all(df, use_interactions=True, group_gds=False, label="FE")

    print("\n[GROUP] GDS con clases agrupadas (5,6,7 -> 'severo')")
    grouped = run_all(df, use_interactions=False, group_gds=True, label="GROUP")

    print("\n[FE+GROUP] GDS con FE + clases agrupadas")
    both = run_all(df, use_interactions=True, group_gds=True, label="FE+GROUP")

    out_dir = comparisons_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in TARGETS:
        for label, results in [("BASE", base), ("FE", with_fe), ("GROUP", grouped), ("FE+GROUP", both)]:
            if target not in results:
                continue
            for r in results[target]:
                rows.append({
                    "experimento": label,
                    "target": target,
                    "modelo": r["model_name"],
                    "n_min": r["n_min"],
                    "k_outer": r["k_outer"],
                    "n_features": r.get("n_features"),
                    "f1_macro_mean": r.get("f1_macro_mean"),
                    "balanced_accuracy_mean": r.get("balanced_accuracy_mean"),
                    "recall_macro_mean": r.get("recall_macro_mean"),
                    "precision_macro_mean": r.get("precision_macro_mean"),
                    "icn": r.get("icn"),
                    "delta_sesgo": r.get("delta_sesgo"),
                })

    df_compare = pd.DataFrame(rows)
    df_compare.to_csv(out_dir / "comparison_baseline_vs_improvements.csv", index=False, encoding="utf-8")
    print(f"\nComparación guardada en {out_dir / 'comparison_baseline_vs_improvements.csv'}")

    print("\n=== RESUMEN: MEJOR F1 POR TARGET Y EXPERIMENTO ===")
    for target in TARGETS:
        print(f"\n{target}:")
        sub = df_compare[df_compare["target"] == target]
        for exp in ["BASE", "FE", "GROUP", "FE+GROUP"]:
            sub_exp = sub[sub["experimento"] == exp]
            if sub_exp.empty:
                continue
            best = sub_exp.loc[sub_exp["f1_macro_mean"].idxmax()]
            print(f"  {exp:8s} | mejor: {best['modelo']:25s} | F1={best['f1_macro_mean']:.4f} | ICN={best['icn']:.3f}")


if __name__ == "__main__":
    main()
