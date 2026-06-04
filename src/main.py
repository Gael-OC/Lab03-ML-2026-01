from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from data_loader import TARGETS, load_sav_dataset, prepare_xy
from evaluation import assign_icn, compute_outer_folds, run_nested_cv, unimplemented_result
from models import build_model_registry
from reports import (
    write_auxiliary_tables,
    write_json_results,
    write_latex_tables,
    write_pdf_tables,
    write_summary_csv,
    write_warnings,
)
from settings import DEFAULT_CONFIG_PATH, ensure_output_dirs, load_config


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Ejecuta el Laboratorio 03 con validacion cruzada anidada.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Ruta al archivo YAML de configuracion.")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Objetivos a ejecutar. Por defecto usa los seis objetivos del PDF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)

    target_names = args.targets or config["experiment"].get("targets", TARGETS)
    model_registry = build_model_registry(random_state=config["experiment"]["outer_random_state"])
    model_order = ["logistic", "svm_linear", "svm_rbf", "tree", "knn"]

    df = load_sav_dataset(config["dataset"]["path"])
    results_by_target = {}

    for target_name in target_names:
        X, y = prepare_xy(df, target_name)
        n_min, k_outer = compute_outer_folds(y, config["experiment"]["max_outer_folds"])
        distribution = {int(label): int(count) for label, count in y.value_counts().sort_index().items()}
        target_results = []

        for model_key in model_order:
            spec = model_registry[model_key]
            if spec.implemented:
                result = run_nested_cv(X, y, target_name, spec, config["experiment"])
            else:
                result = unimplemented_result(target_name, spec, distribution, n_min, k_outer)
            target_results.append(result)

        assign_icn(target_results)
        results_by_target[target_name] = target_results

    output_dirs = {name: Path(path) for name, path in config["outputs"].items()}
    tables_dir = output_dirs["tables"]

    write_summary_csv(results_by_target, tables_dir / "resumen_resultados.csv")
    write_json_results(results_by_target, tables_dir / "resultados_detallados.json")
    write_auxiliary_tables(results_by_target, output_dirs)
    write_warnings(results_by_target, output_dirs["root"] / "advertencias.txt")
    write_latex_tables(results_by_target, tables_dir / "resultados_experimentos.tex")
    write_pdf_tables(results_by_target, tables_dir / "resultados_experimentos.pdf")

    print("Experimentos finalizados.")
    print(f"Tabla LaTeX: {tables_dir / 'resultados_experimentos.tex'}")
    print(f"Tabla PDF:   {tables_dir / 'resultados_experimentos.pdf'}")


if __name__ == "__main__":
    main()
