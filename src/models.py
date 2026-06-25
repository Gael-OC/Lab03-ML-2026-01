from dataclasses import dataclass
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    implemented: bool
    pipeline: Pipeline | None
    param_grid: dict[str, list[Any]] | None
    student_note: str = ""


MODEL_ORDER: list[str] = [
    "dummy",
    "logistic",
    "svm_linear",
    "svm_rbf",
    "tree",
    "knn",
]
"""Orden canonico de los modelos en tablas, plots y tests pareados."""

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "dummy": "Baseline (Dummy)",
    "logistic": "Regresion Logistica",
    "svm_linear": "SVM lineal",
    "svm_rbf": "SVM RBF",
    "tree": "Arbol de decision",
    "knn": "K-NN",
}
"""Mapeo de model_key a nombre legible."""


def build_model_registry(random_state: int = 42) -> dict[str, ModelSpec]:
    """Registro de los seis modelos que compara el laboratorio.

    Incluye ``dummy`` como linea base trivial que predice respetando
    la distribucion marginal de clases. Sirve para responder: ¿que
    tan bueno es un F1 de 0.33 en GDS comparado con adivinar?

    Ademas, los cinco clasificadores pedidos por el enunciado:
    Regresion Logistica, SVM lineal, SVM RBF, Arbol de Decision y K-NN.
    """
    logistic = ModelSpec(
        key="logistic",
        display_name="Regresion Logistica",
        implemented=True,
        pipeline=Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        param_grid={"clf__C": [0.01, 0.1, 1, 10]},
    )

    return {
        "dummy": ModelSpec(
            key="dummy",
            display_name="Baseline (DummyClassifier)",
            implemented=True,
            pipeline=Pipeline([
                ("clf", DummyClassifier(strategy="stratified", random_state=random_state)),
            ]),
            param_grid={},
        ),
        logistic.key: logistic,
        "svm_linear": ModelSpec(
            key="svm_linear",
            display_name="SVM lineal",
            implemented=True,
            # SVC ignora random_state salvo cuando probability=True
            # (que no usamos). No se pasa para no sugerir una
            # determinismo que no existe; la semilla efectiva
            # viene del StratifiedKFold externo (random_state=42).
            pipeline=Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", class_weight="balanced")),
            ]),
            param_grid={"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "svm_rbf": ModelSpec(
            key="svm_rbf",
            display_name="SVM RBF",
            implemented=True,
            # Misma justificacion que SVM lineal arriba.
            pipeline=Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", class_weight="balanced")),
            ]),
            param_grid={
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.01, 0.1, 1],
            },
        ),
        "tree": ModelSpec(
            key="tree",
            display_name="Arbol de decision",
            implemented=True,
            pipeline=Pipeline([
                ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=random_state)),
            ]),
            param_grid={
                "clf__max_depth": [2, 3, 4, 5, None],
                "clf__min_samples_leaf": [1, 3, 5, 10],
            },
        ),
        "knn": ModelSpec(
            key="knn",
            display_name="K-NN",
            implemented=True,
            pipeline=Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            param_grid={
                "clf__n_neighbors": [3, 5, 7, 9, 11],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"],
            },
        ),
    }

