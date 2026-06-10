from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from analysis import run_advanced_analysis
from data_loader import TARGETS, load_sav_dataset, prepare_xy
from eda import run_eda
from evaluation import assign_icn, compute_delta_sesgo, compute_outer_folds, run_nested_cv, unimplemented_result
from models import build_model_registry
from plots import generate_all_plots
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
    parser.add_argument("--skip-eda", action="store_true", help="Salta el analisis exploratorio.")
    parser.add_argument("--skip-analysis", action="store_true", help="Salta el analisis avanzado.")
    parser.add_argument("--skip-plots", action="store_true", help="Salta las visualizaciones.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)

    df = load_sav_dataset(config["dataset"]["path"])

    output_dirs = {name: Path(path) for name, path in config["outputs"].items()}

    # EDA
    if not args.skip_eda:
        eda_dir = output_dirs.get("figures") / "eda" if "figures" in output_dirs else output_dirs["tables"] / ".." / "figures" / "eda"
        eda_dir.mkdir(parents=True, exist_ok=True)
        run_eda(df, eda_dir)

    # Experimentos
    target_names = args.targets or config["experiment"].get("targets", TARGETS)
    model_registry = build_model_registry(random_state=config["experiment"]["outer_random_state"])
    model_order = ["logistic", "svm_linear", "svm_rbf", "tree", "knn"]

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
        compute_delta_sesgo(target_results)
        results_by_target[target_name] = target_results

    tables_dir = output_dirs["tables"]

    write_summary_csv(results_by_target, tables_dir / "resumen_resultados.csv")
    write_json_results(results_by_target, tables_dir / "resultados_detallados.json")
    write_auxiliary_tables(results_by_target, output_dirs)
    write_warnings(results_by_target, output_dirs["root"] / "advertencias.txt")
    write_latex_tables(results_by_target, tables_dir / "resultados_experimentos.tex")
    write_pdf_tables(results_by_target, tables_dir / "resultados_experimentos.pdf")

    # Analisis avanzado
    if not args.skip_analysis:
        analysis_dir = output_dirs.get("figures") / "analysis" if "figures" in output_dirs else output_dirs["tables"] / ".." / "figures" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        run_advanced_analysis(results_by_target, df, output_dirs)

    # Visualizaciones
    if not args.skip_plots:
        plots_dir = output_dirs.get("figures") / "experiments" if "figures" in output_dirs else output_dirs["tables"] / ".." / "figures" / "experiments"
        plots_dir.mkdir(parents=True, exist_ok=True)
        generate_all_plots(results_by_target, plots_dir)

    print("Experimentos finalizados.")
    print(f"Tabla LaTeX: {tables_dir / 'resultados_experimentos.tex'}")
    print(f"Tabla PDF:   {tables_dir / 'resultados_experimentos.pdf'}")


if __name__ == "__main__":
    main()
