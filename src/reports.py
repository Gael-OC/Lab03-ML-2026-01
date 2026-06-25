from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


SUMMARY_COLUMNS = [
    "target",
    "model_name",
    "status",
    "n_min",
    "k_outer",
    "f1_macro_mean",
    "f1_macro_std",
    "balanced_accuracy_mean",
    "recall_macro_mean",
    "precision_macro_mean",
    "stability_raw",
    "stability",
    "icn_raw",
    "icn",
    "delta_sesgo",
    "best_params_mode",
    "message",
]


def write_summary_csv(results_by_target: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    rows = [item for results in results_by_target.values() for item in results]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in SUMMARY_COLUMNS})


def write_json_results(results_by_target: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results_by_target, handle, ensure_ascii=False, indent=2)


def write_auxiliary_tables(results_by_target: dict[str, list[dict[str, Any]]], output_dirs: dict[str, Path]) -> None:
    distributions_path = output_dirs["tables"] / "distribucion_clases.csv"
    with distributions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "class", "support"])
        for target, results in results_by_target.items():
            distribution = results[0]["class_distribution"]
            for label, support in distribution.items():
                writer.writerow([target, label, support])

    for target, results in results_by_target.items():
        for item in results:
            if not item["implemented"]:
                continue
            _write_confusion_matrix(target, item, output_dirs["confusion_matrices"])
            _write_class_report(target, item, output_dirs["per_class"])


def write_warnings(results_by_target: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    lines: list[str] = []
    for target, results in results_by_target.items():
        for item in results:
            for warning in item.get("warnings", []):
                lines.append(f"[{target} | {item['model_name']}] {warning}")
    if not lines:
        lines.append("No se registraron advertencias.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_tables(results_by_target: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    lines = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[spanish]{babel}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\geometry{margin=1.5cm, landscape}",
        r"\begin{document}",
        r"\section*{Laboratorio 03: resultados de validacion cruzada anidada}",
        (
            "Los 5 clasificadores est\\'an implementados. "
            "Se reportan promedios y desviaciones sobre los folds externos."
        ),
        "",
    ]

    for target, results in results_by_target.items():
        distribution = _format_distribution(results[0]["class_distribution"])
        lines.extend(
            [
                rf"\subsection*{{Experimento {latex_escape(target)}}}",
                rf"\noindent\textbf{{Distribuci\'on de clases:}} {latex_escape(distribution)}. "
                rf"\textbf{{n\_min:}} {results[0]['n_min']}. "
                rf"\textbf{{k externo:}} {results[0]['k_outer']}.",
                r"\begin{table}[h]",
                r"\centering",
                rf"\caption{{Resultados para {latex_escape(target)}}}",
                r"\scriptsize",
                r"\begin{tabular}{p{2.5cm}p{1.8cm}p{1.4cm}p{1.3cm}p{1.4cm}p{1.0cm}p{1.0cm}p{1.0cm}p{1.0cm}p{2.5cm}}",
                r"\toprule",
                r"Modelo & F1 macro & BalAcc & Recall & Precision & Estab* & ICN* & ICN & $\Delta$sesgo & Hiperparámetros / estado \\",
                r"\midrule",
            ]
        )
        for item in results:
            lines.append(_latex_row(item))
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )

    lines.extend([r"\end{document}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_pdf_tables(
    results_by_target: dict[str, list[dict[str, Any]]],
    output_path: Path,
    figures_dir: Path | None = None,
) -> None:
    """Genera un PDF con tablas de metricas y, opcionalmente, figuras.

    Usa reportlab, que soporta fuentes TrueType con caracteres
    acentuados, imagenes raster y tablas con estilos consistentes.

    Si ``figures_dir`` apunta a un directorio que contiene
    ``f1_macro_heatmap.png`` y ``confusion_matrices.png`` (los
    archivos que produce ``plots.py``), se incluyen como figuras.
    """
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.fontSize = 9
    body_style.leading = 12

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Laboratorio 03: resultados de validacion cruzada anidada",
    )

    story: list = []
    story.append(Paragraph("Laboratorio 03: resultados", title_style))
    story.append(Paragraph(
        "Validacion cruzada anidada con grillas reducidas sobre los seis "
        "experimentos del laboratorio. Se reportan promedios y desviaciones "
        "sobre los folds externos, ademas del ICN crudo (formula del PDF, "
        "comparable entre targets) y el ICN normalizado (util para ordenar "
        "modelos dentro de un mismo target).",
        body_style,
    ))
    story.append(Spacer(1, 0.5 * cm))

    has_figure = figures_dir is not None and (figures_dir / "f1_macro_heatmap.png").exists()

    if has_figure:
        story.append(Paragraph("F1 macro por modelo y objetivo", h2_style))
        story.append(Image(str(figures_dir / "f1_macro_heatmap.png"), width=22 * cm, height=8 * cm))
        story.append(Spacer(1, 0.5 * cm))

    for target, results in results_by_target.items():
        story.append(Paragraph(f"Experimento {target}", h2_style))
        distribution = _format_distribution(results[0]["class_distribution"])
        n_min = results[0]["n_min"]
        k_outer = results[0]["k_outer"]
        story.append(Paragraph(
            f"Distribucion: {distribution}. n_min = {n_min}. k externo = {k_outer}.",
            body_style,
        ))
        story.append(Spacer(1, 0.3 * cm))

        story.append(_build_reportlab_table(results))
        story.append(PageBreak())

    cm_figure = figures_dir / "confusion_matrices.png" if figures_dir else None
    if cm_figure is not None and cm_figure.exists():
        story.append(Paragraph("Matrices de confusion (mejor modelo por target)", h2_style))
        story.append(Image(str(cm_figure), width=22 * cm, height=12 * cm))

    doc.build(story)


def _build_reportlab_table(results: list[dict[str, Any]]):
    """Tabla con la misma informacion que las tablas LaTeX."""
    header = [
        "Modelo", "F1 macro", "BalAcc", "Recall", "Precision",
        "Estab*", "ICN*", "ICN", "Delta sesgo", "Hiperparametros",
    ]
    rows: list[list[str]] = [header]
    for item in results:
        if not item["implemented"]:
            rows.append([item["model_name"], "No implementado", *[""] * 8])
            continue
        rows.append([
            item["model_name"],
            _format_mean_std(item["f1_macro_mean"], item["f1_macro_std"]),
            _format_float(item["balanced_accuracy_mean"]),
            _format_float(item["recall_macro_mean"]),
            _format_float(item["precision_macro_mean"]),
            _format_float(item["stability_raw"]),
            _format_float(item["icn_raw"]),
            _format_float(item["icn"]),
            _format_float(item.get("delta_sesgo")),
            item["best_params_mode"],
        ])

    table = Table(
        rows,
        colWidths=[3.4 * cm, 2.6 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm,
                   1.5 * cm, 1.5 * cm, 1.5 * cm, 1.8 * cm, 6.0 * cm],
        repeatRows=1,
    )
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (1, 1), (-2, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
    ])
    for row_idx in range(1, len(rows)):
        if row_idx % 2 == 0:
            style.add("BACKGROUND", (0, row_idx), (-1, row_idx), colors.whitesmoke)
    table.setStyle(style)
    return table


def _write_confusion_matrix(target: str, item: dict[str, Any], output_dir: Path) -> None:
    labels = item["labels"]
    matrix = item["confusion_matrix"]
    path = output_dir / f"matriz_confusion_{target}_{item['model_key']}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["real/predicho", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _write_class_report(target: str, item: dict[str, Any], output_dir: Path) -> None:
    report = item["classification_report"]
    path = output_dir / f"metricas_por_clase_{target}_{item['model_key']}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for label in item["labels"]:
            row = report[str(label)]
            writer.writerow(
                [
                    label,
                    _format_float(row["precision"]),
                    _format_float(row["recall"]),
                    _format_float(row["f1-score"]),
                    int(row["support"]),
                ]
            )


def _latex_row(item: dict[str, Any]) -> str:
    model = latex_escape(item["model_name"])
    if not item["implemented"]:
        status = latex_escape(item["status"])
        return (
            f"{model} & {status} & {status} & {status} & {status} & "
            f"{status} & {status} & {status} & {latex_escape(item['message'])} \\\\"
        )

    return (
        f"{model} & "
        f"{_format_mean_std(item['f1_macro_mean'], item['f1_macro_std'])} & "
        f"{_format_float(item['balanced_accuracy_mean'])} & "
        f"{_format_float(item['recall_macro_mean'])} & "
        f"{_format_float(item['precision_macro_mean'])} & "
        f"{_format_float(item['stability_raw'])} & "
        f"{_format_float(item['icn_raw'])} & "
        f"{_format_float(item['icn'])} & "
        f"{_format_float(item.get('delta_sesgo'))} & "
        f"{latex_escape(item['best_params_mode'])} \\\\"
    )


def _format_distribution(distribution: dict[int, int]) -> str:
    return ", ".join(f"{label}: {support}" for label, support in distribution.items())


def _format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "No implementado"
    return f"{mean:.3f} +/- {std:.3f}"


def _format_float(value: float | None) -> str:
    if value is None:
        return "No implementado"
    return f"{value:.3f}"


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return value


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)
