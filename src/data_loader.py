from pathlib import Path
from itertools import combinations

import pandas as pd
import pyreadstat


FEATURE_COLS = [
    "Día",
    "Mes",
    "Año",
    "Estación",
    "País",
    "Ciudad",
    "CalleLugar",
    "NumeroPiso",
    "Miguel2",
    "González2",
    "Avenida2",
    "Imperial2",
    "A682",
    "Caldera2",
    "Copiapo2",
]

TARGETS = ["GDS", "GDS_R1", "GDS_R2", "GDS_R3", "GDS_R4", "GDS_R5"]

ORIENT_TEMPORAL = ["Día", "Mes", "Año", "Estación"]
ORIENT_ESPACIAL = ["País", "Ciudad", "CalleLugar", "NumeroPiso"]
MEMORIA_VERBAL = ["Miguel2", "González2", "Avenida2", "Imperial2"]
MEMORIA_GEOGRAFICA = ["A682", "Caldera2", "Copiapo2"]


def _interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Genera features de interacción AND entre atributos del mismo grupo semántico."""
    groups = [ORIENT_TEMPORAL, ORIENT_ESPACIAL, MEMORIA_VERBAL, MEMORIA_GEOGRAFICA]
    out = pd.DataFrame(index=df.index)
    for group in groups:
        for col_a, col_b in combinations(group, 2):
            name = f"{col_a[:3]}_x_{col_b[:3]}"
            out[name] = (df[col_a].astype(int) & df[col_b].astype(int)).astype(float)
    return out


def get_feature_cols(use_interactions: bool = False) -> list[str]:
    """Devuelve la lista de features a usar en X.

    Si use_interactions=True, incluye 21 features de interacción AND
    entre pares de atributos del mismo grupo semántico.
    """
    if not use_interactions:
        return list(FEATURE_COLS)
    interactions = list(_interactions_columns())
    return list(FEATURE_COLS) + interactions


def _interactions_columns() -> list[str]:
    groups = [ORIENT_TEMPORAL, ORIENT_ESPACIAL, MEMORIA_VERBAL, MEMORIA_GEOGRAFICA]
    cols = []
    for group in groups:
        for col_a, col_b in combinations(group, 2):
            cols.append(f"{col_a[:3]}_x_{col_b[:3]}")
    return cols


def load_sav_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset configurado: {path}")

    df, _ = pyreadstat.read_sav(str(path))
    _validate_columns(df)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    required_cols = ["ID", *FEATURE_COLS, *TARGETS]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas en el .sav: {missing_cols}")


def prepare_xy(
    df: pd.DataFrame, target_name: str, use_interactions: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    if target_name not in TARGETS:
        raise ValueError(f"Objetivo no reconocido: {target_name}")

    X = df[FEATURE_COLS].astype(float)
    if use_interactions:
        interactions = _interactions(df)
        X = pd.concat([X, interactions], axis=1)
    y = df[target_name].astype(int)
    return X, y


def group_rare_gds_classes(df: pd.DataFrame, threshold: int = 50) -> pd.DataFrame:
    """Agrupa clases raras de GDS (con soporte < threshold) en una clase 'severo'.

    Estrategia: cualquier clase con soporte < threshold se agrupa en una
    unica clase 'severo' usando el mayor label + 1. Asi, las clases raras
    contiguas en la escala clinica quedan juntas y las estables se preservan.
    """
    df = df.copy()
    counts = df["GDS"].value_counts()
    rare_classes = counts[counts < threshold].index.tolist()
    if not rare_classes:
        return df

    new_label = int(counts.index.max()) + 1
    df.loc[df["GDS"].isin(rare_classes), "GDS"] = new_label
    return df
