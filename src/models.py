from dataclasses import dataclass
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    implemented: bool
    pipeline: Pipeline | None
    param_grid: dict[str, list[Any]] | None
    student_note: str = ""


def build_model_registry(random_state: int = 42) -> dict[str, ModelSpec]:
    logistic = ModelSpec(
        key="logistic",
        display_name="Regresión Logística",
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
        logistic.key: logistic,
        "svm_linear": ModelSpec(
            key="svm_linear",
            display_name="SVM lineal",
            implemented=False,
            pipeline=None,
            param_grid=None,
            student_note="Pendiente: implementar Pipeline con StandardScaler y SVC(kernel='linear').",
        ),
        "svm_rbf": ModelSpec(
            key="svm_rbf",
            display_name="SVM RBF",
            implemented=False,
            pipeline=None,
            param_grid=None,
            student_note="Pendiente: implementar Pipeline con StandardScaler y SVC(kernel='rbf').",
        ),
        "tree": ModelSpec(
            key="tree",
            display_name="Árbol de decisión",
            implemented=False,
            pipeline=None,
            param_grid=None,
            student_note="Pendiente: implementar DecisionTreeClassifier con regularizacion estructural.",
        ),
        "knn": ModelSpec(
            key="knn",
            display_name="K-NN",
            implemented=False,
            pipeline=None,
            param_grid=None,
            student_note="Pendiente: implementar Pipeline con StandardScaler y KNeighborsClassifier.",
        ),
    }


# Plantillas para los alumnos. Estan comentadas a proposito: el laboratorio
# solicita implementar solo Regresion Logistica en esta entrega base.
#
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
#
# "svm_linear": ModelSpec(
#     key="svm_linear",
#     display_name="SVM lineal",
#     implemented=True,
#     pipeline=Pipeline([
#         ("scaler", StandardScaler()),
#         ("clf", SVC(kernel="linear", class_weight="balanced")),
#     ]),
#     param_grid={"clf__C": [0.01, 0.1, 1, 10]},
# )
#
# "svm_rbf": ModelSpec(
#     key="svm_rbf",
#     display_name="SVM RBF",
#     implemented=True,
#     pipeline=Pipeline([
#         ("scaler", StandardScaler()),
#         ("clf", SVC(kernel="rbf", class_weight="balanced")),
#     ]),
#     param_grid={
#         "clf__C": [0.1, 1, 10],
#         "clf__gamma": ["scale", 0.01, 0.1, 1],
#     },
# )
#
# "tree": ModelSpec(
#     key="tree",
#     display_name="Arbol de decision",
#     implemented=True,
#     pipeline=Pipeline([
#         ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42)),
#     ]),
#     param_grid={
#         "clf__max_depth": [2, 3, 4, 5, None],
#         "clf__min_samples_leaf": [1, 3, 5, 10],
#     },
# )
#
# "knn": ModelSpec(
#     key="knn",
#     display_name="K-NN",
#     implemented=True,
#     pipeline=Pipeline([
#         ("scaler", StandardScaler()),
#         ("clf", KNeighborsClassifier()),
#     ]),
#     param_grid={
#         "clf__n_neighbors": [3, 5, 7, 9, 11],
#         "clf__weights": ["uniform", "distance"],
#         "clf__metric": ["euclidean", "manhattan"],
#     },
# )
