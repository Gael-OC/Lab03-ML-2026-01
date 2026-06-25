# Laboratorio 03 - Machine Learning

**Grupo**: 1
**Estudiantes**: Gael Ortega y Matías Vidal

Este proyecto corresponde al Laboratorio 03 de Machine Learning. El objetivo principal es construir, ajustar y comparar cinco clasificadores fundamentales (Regresión Logística, SVM lineal, SVM con kernel RBF, Árbol de Decisión y K-NN) sobre seis variables objetivo distintas para predecir niveles de deterioro cognitivo a partir de 15 atributos binarios extraídos de un test neuropsicológico reducido.

El laboratorio implementa validación cruzada anidada, ajuste de hiperparámetros con grillas reducidas, comparación mediante métricas macro, y un análisis posterior con feature engineering, tests de significancia, bootstrap y una comparación contra un baseline trivial.

---

## 1. Dataset utilizado

El dataset es un archivo `.sav` de SPSS con 1119 observaciones. Cada fila corresponde a la respuesta de un paciente a un test neuropsicológico reducido. Las columnas son:

```text
ID              -> identificador (no se usa como feature)
15 features     -> respuestas binarias (0 = fallo, 1 = acierto)
6 objetivos     -> GDS, GDS_R1, GDS_R2, GDS_R3, GDS_R4, GDS_R5
```

Los 15 atributos binarios se agrupan en cuatro dimensiones cognitivas:

```text
Orientación temporal -> Día, Mes, Año, Estación
Orientación espacial -> País, Ciudad, CalleLugar, NumeroPiso
Memoria verbal       -> Miguel2, González2, Avenida2, Imperial2
Memoria geográfica   -> A682, Caldera2, Copiapo2
```

Las seis columnas objetivo corresponden a diferentes formas de agrupar los niveles de la escala GDS (Global Deterioration Scale):

```text
GDS    -> escala original (7 niveles)
GDS_R1 -> reagrupación en 3 niveles
GDS_R2 -> reagrupación en 3 niveles
GDS_R3 -> reagrupación binaria
GDS_R4 -> reagrupación en 3 niveles
GDS_R5 -> reagrupación en 3 niveles
```

Cada columna objetivo se evalúa como un experimento independiente.

---

## 2. Estructura general del proyecto

```text
.
├── config/
│   └── paths.yaml
├── datasets/
│   └── 15 atributos R0-R5.sav
├── src/
│   ├── __init__.py
│   ├── settings.py
│   ├── data_loader.py
│   ├── models.py
│   ├── evaluation.py
│   ├── significance.py
│   ├── reports.py
│   ├── eda.py
│   ├── analysis.py
│   ├── plots.py
│   ├── compare_improvements.py
│   ├── plot_comparisons.py
│   └── main.py
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── confusion_matrices/
│   ├── per_class/
│   ├── comparisons/
│   └── advertencias.txt
├── environment.yml
└── README.md
```

Descripción de los archivos principales:

| Archivo | Descripción |
|---|---|
| `src/main.py` | Ejecuta el flujo completo: EDA, experimentos, análisis y plots. |
| `src/data_loader.py` | Carga el `.sav`, valida columnas y entrega X, y. |
| `src/models.py` | Define los seis clasificadores con sus `Pipeline` y grillas. |
| `src/evaluation.py` | Implementa la validación cruzada anidada y las métricas. |
| `src/significance.py` | Implementa Wilcoxon pareado y McNemar con corrección de Yates. |
| `src/reports.py` | Genera tablas CSV, LaTeX y PDF (con reportlab). |
| `src/eda.py` | Genera las figuras del análisis exploratorio. |
| `src/analysis.py` | Curvas de aprendizaje, feature importance, bootstrap, tests. |
| `src/plots.py` | Genera los heatmaps y figuras de comparación. |
| `src/compare_improvements.py` | Compara el baseline contra feature engineering y agrupación. |
| `config/paths.yaml` | Configuración de rutas y parámetros del experimento. |

---

## 3. Preprocesamiento aplicado

El flujo de preprocesamiento es:

```text
.sav de SPSS
-> DataFrame pandas
-> 15 features binarias (float)
-> target como entero
-> StandardScaler dentro del Pipeline
```

Como las features son binarias, no se aplicó imputación de missings ni codificación adicional. El escalado se hace **dentro del Pipeline** con `StandardScaler` para que se ajuste solo con los datos de entrenamiento de cada fold, evitando fuga de información.

**Excepciones:**

- El Árbol de Decisión **no** usa `StandardScaler` porque es invariante a transformaciones monótonas.
- K-NN usa distancia Manhattan o Euclidiana. Para datos binarios, la distancia Manhattan es proporcional a la distancia Hamming (cuenta bits diferentes), lo que tiene una justificación clara.
- El baseline `DummyClassifier` no usa scaler ni features (predice según la distribución marginal).

---

## 4. Modelos implementados

Se implementaron los seis clasificadores que compara el laboratorio, cada uno dentro de un `Pipeline` con `StandardScaler` (excepto el árbol y el baseline):

```text
DummyClassifier         -> DummyClassifier(strategy="stratified")  (baseline)
Regresión Logística     -> StandardScaler + LogisticRegression(class_weight="balanced")
SVM lineal              -> StandardScaler + SVC(kernel="linear", class_weight="balanced")
SVM con kernel RBF      -> StandardScaler + SVC(kernel="rbf", class_weight="balanced")
Árbol de decisión       -> DecisionTreeClassifier(class_weight="balanced")
K-NN                    -> StandardScaler + KNeighborsClassifier
```

Todos los modelos que lo soportan usan `class_weight="balanced"` para mitigar el desbalance severo de clases. K-NN no tiene este parámetro, pero `weights="distance"` ayuda a compensar.

### Grillas de hiperparámetros

```text
Regresión Logística  -> C = [0.01, 0.1, 1, 10]
SVM lineal           -> C = [0.01, 0.1, 1, 10]
SVM con kernel RBF   -> C = [0.1, 1, 10], gamma = [scale, 0.01, 0.1, 1]
Árbol de decisión    -> max_depth = [2, 3, 4, 5, None], min_samples_leaf = [1, 3, 5, 10]
K-NN                 -> n_neighbors = [3, 5, 7, 9, 11], weights = [uniform, distance], metric = [euclidean, manhattan]
DummyClassifier      -> sin hiperparámetros
```

Las grillas son reducidas a propósito, para reducir el riesgo de selección optimista de hiperparámetros en un dataset con tan pocas muestras por clase.

---

## 5. Validación cruzada anidada

El laboratorio usa **validación cruzada anidada** para separar correctamente la selección de hiperparámetros de la evaluación final:

```text
Ciclo externo (k_outer) -> estima el desempeño final
Ciclo interno (k_inner) -> selecciona hiperparámetros con GridSearchCV
```

El conjunto de prueba externo nunca se usa para elegir hiperparámetros ni para reportar resultados parciales.

### Número de folds

El número de folds se adapta al soporte de la clase menos representada:

```text
k_outer = min(5, n_min)
k_inner = min(3, k_outer)
```

| Target   | n_clases | n_min | k_outer | Ratio de desbalance |
|---|---:|---:|---:|---:|
| GDS      | 7 | 2   | 2 | 250.0 |
| GDS_R1   | 3 | 22  | 5 | 43.0  |
| GDS_R2   | 3 | 172 | 5 | 3.8   |
| GDS_R3   | 2 | 172 | 5 | 5.5   |
| GDS_R4   | 3 | 64  | 5 | 14.2  |
| GDS_R5   | 3 | 149 | 5 | 5.4   |

El target `GDS` es el más difícil porque tiene 7 clases muy desbalanceadas (la clase 7 tiene solo 2 muestras), lo que obliga a usar `k_outer = 2`. En este caso el ciclo interno usa `KFold` no estratificado, lo que queda registrado en `outputs/advertencias.txt`.

---

## 6. Métricas utilizadas

Para todos los modelos se calcularon las siguientes métricas, promediando sobre los folds externos:

```text
accuracy
balanced_accuracy
precision_macro
recall_macro
f1_macro
```

La métrica principal de selección interna es `f1_macro`, porque en problemas desbalanceados la accuracy es engañosa. Adicionalmente se calculan:

```text
ICN (Índice Comparativo Normalizado)   -> resumen ponderado de 5 métricas
Δsesgo                                  -> diferencia entre score interno y externo
Bootstrap CI 95%                        -> intervalo de confianza por bootstrap
Wilcoxon + McNemar-Yates                -> significancia estadística entre pares
```

### Fórmula del ICN

El ICN del enunciado es una suma ponderada de métricas crudas, comparable entre targets:

```text
ICN = 0.40·F1_macro + 0.25·BA + 0.20·Recall_macro + 0.10·Precision_macro + 0.05·Estabilidad
```

donde cada métrica está en su escala cruda (no normalizada), por lo que el ICN es comparable entre los seis targets. Adicionalmente el laboratorio reporta una versión normalizada min-max entre los modelos de un mismo target, útil para ordenar modelos dentro del target.

---

## 7. Instalación y preparación

El proyecto utiliza un entorno de Conda definido en `environment.yml`.

Crear el entorno:

```bash
conda env create -f environment.yml
```

Activar el entorno:

```bash
conda activate lab03_ml_2026_01
```

El dataset debe ubicarse en:

```text
datasets/15 atributos R0-R5.sav
```

Si se mueve, ajustar la ruta en `config/paths.yaml`.

---

## 8. Ejecución del laboratorio

Para ejecutar el flujo completo (EDA + 36 experimentos + análisis + plots + comparaciones):

```bash
python src/main.py --keep-estimators
```

La opción `--keep-estimators` persiste los estimadores de cada fold en `outputs/estimator_cache/`, lo que permite calcular los tests de significancia sin reentrenar la validación anidada. Sin esta opción, los tests de significancia se omiten (no hay cache que leer).

También se puede ejecutar solo un subconjunto de objetivos:

```bash
python src/main.py --targets GDS_R3
```

O saltar partes del flujo:

```bash
python src/main.py --skip-eda
python src/main.py --skip-analysis
python src/main.py --skip-plots
```

Para correr la comparación contra las mejoras (feature engineering y agrupación de clases raras):

```bash
python src/compare_improvements.py
python src/plot_comparisons.py
```

---

## 9. Análisis exploratorio (EDA)

El EDA genera cuatro figuras y una tabla resumen en `outputs/figures/eda/`:

```text
class_distribution.png       -> distribución de clases por target con n_min y k_outer
feature_correlation.png      -> matriz de correlación (Pearson = phi para binarias)
target_relationships.png     -> cómo GDS_Rk se derivan de GDS
eda_summary.csv              -> tabla resumen por target
```

Hallazgos principales del EDA:

- El dataset no tiene valores faltantes ni filas duplicadas.
- `GDS` tiene 7 clases con razón de desbalance de 250:1, lo que lo hace intrínsecamente difícil.
- `GDS_R3` es el problema más simple (binario, clases 1 y 3) y también el más fácil de predecir.
- Los atributos de orientación temporal están altamente correlacionados entre sí (acertar Día, Mes, Año, Estación suele ser un "paquete").

---

## 10. Resultados por experimento

Todos los valores se calcularon con `python src/main.py --keep-estimators`.

### 10.1 GDS (escala original, 7 clases, k=2)

| Modelo | F1 macro | Balanced Acc. | ICN | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.140 | 0.139 | 0.182 | default |
| Regresión Logística | 0.295 ± 0.038 | 0.440 | 0.395 | `C=0.01` |
| SVM lineal | 0.308 ± 0.023 | 0.470 | 0.415 | `C=0.01` |
| SVM RBF | 0.335 ± 0.039 | 0.452 | 0.419 | `C=10, gamma=0.1` |
| Árbol de decisión | 0.223 ± 0.033 | 0.412 | 0.348 | `max_depth=5, min_leaf=5` |
| K-NN | 0.317 ± 0.033 | 0.327 | 0.357 | `manhattan, k=3, uniform` |

SVM RBF gana en ICN. Sin embargo, todos los modelos están por debajo de 0.34 de F1 macro, lo que muestra que `GDS` con 7 clases y 2 muestras en la clase más rara es un problema muy difícil. El baseline (0.140) confirma que los modelos sí están aprendiendo por encima de predecir según la distribución marginal.

**Nota:** por su k_outer=2, este caso se reporta como descriptivo. Los tests de significancia no son informativos con tan pocos folds.

### 10.2 GDS_R1 (3 clases, k=5)

| Modelo | F1 macro | Balanced Acc. | ICN | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.337 | 0.334 | 0.368 | default |
| Regresión Logística | 0.651 ± 0.063 | 0.765 | 0.712 | `C=0.01` |
| SVM lineal | 0.654 ± 0.056 | 0.770 | 0.716 | `C=0.01` |
| SVM RBF | 0.671 ± 0.058 | 0.781 | 0.729 | `C=0.1, gamma=0.01` |
| Árbol de decisión | 0.592 ± 0.051 | 0.710 | 0.662 | `max_depth=3, min_leaf=1` |
| K-NN | 0.689 ± 0.064 | 0.676 | 0.700 | `euclidean, k=11, distance` |

K-NN tiene el F1 macro más alto (0.689), pero SVM RBF gana en ICN (0.729) y en balanced accuracy (0.781). La diferencia entre ambos no es estadísticamente significativa según McNemar-Yates.

### 10.3 GDS_R2 (3 clases, k=5, n_min=172)

| Modelo | F1 macro | Balanced Acc. | ICN | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.312 | 0.312 | 0.345 | default |
| Regresión Logística | 0.663 ± 0.043 | 0.666 | 0.679 | `C=0.01` |
| SVM lineal | 0.667 ± 0.048 | 0.662 | 0.681 | `C=0.01` |
| SVM RBF | 0.675 ± 0.056 | 0.667 | 0.687 | `C=0.1, gamma=0.01` |
| Árbol de decisión | 0.648 ± 0.032 | 0.658 | 0.669 | `max_depth=4, min_leaf=1` |
| K-NN | 0.635 ± 0.034 | 0.618 | 0.648 | `euclidean, k=7, uniform` |

Este es el target con el desbalance más suave (ratio 3.8:1). SVM RBF lidera por ICN, pero la diferencia con SVM lineal (0.681) es mínima. McNemar-Yates no detecta diferencias significativas en este target.

### 10.4 GDS_R3 (binario, k=5)

| Modelo | F1 macro | Balanced Acc. | ICN | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.499 | 0.499 | 0.522 | default |
| Regresión Logística | 0.764 ± 0.031 | 0.842 | 0.806 | `C=0.01` |
| SVM lineal | 0.757 ± 0.043 | 0.836 | 0.800 | `C=0.01` |
| SVM RBF | 0.792 ± 0.039 | 0.838 | 0.818 | `C=0.1, gamma=0.01` |
| Árbol de decisión | 0.733 ± 0.077 | 0.795 | 0.771 | `max_depth=2, min_leaf=1` |
| K-NN | 0.791 ± 0.033 | 0.764 | 0.792 | `euclidean, k=11, uniform` |

GDS_R3 es binario y el problema más "fácil" del laboratorio. SVM RBF y K-NN empatan en F1 macro (0.792), y SVM RBF gana en ICN. El baseline (0.499) muestra que el problema es casi adivinar al ser binario y balanceado, pero los modelos reales ganan por +0.30.

### 10.5 GDS_R4 (3 clases, k=5, n_min=64)

| Modelo | F1 macro | Balanced Acc. | ICN | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.302 | 0.301 | 0.335 | default |
| Regresión Logística | 0.532 ± 0.015 | 0.718 | 0.637 | `C=0.01` |
| SVM lineal | 0.521 ± 0.025 | 0.709 | 0.629 | `C=0.01` |
| SVM RBF | 0.550 ± 0.039 | 0.590 | 0.591 | `C=0.1, gamma=1` |
| Árbol de decisión | 0.515 ± 0.027 | 0.659 | 0.603 | `max_depth=None, min_leaf=1` |
| K-NN | 0.601 ± 0.042 | 0.593 | 0.619 | `euclidean, k=3, uniform` |

Este es el único target donde **Regresión Logística gana en ICN** (0.637). K-NN tiene el F1 macro más alto, pero su balanced accuracy es la peor del lote (0.593), lo que indica que ignora la clase minoritaria. SVM RBF también gana en F1 entre los modelos "complejos" pero con balanced accuracy baja (0.590). Regresión Logística con `class_weight="balanced"` obtiene F1=0.532 y balanced accuracy=0.718, el balance más alto del lote.

### 10.6 GDS_R5 (3 clases, k=5, n_min=149)

| Modelo | F1 macro | Balanced Acc. | ICN | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.327 | 0.325 | 0.360 | default |
| Regresión Logística | 0.521 ± 0.029 | 0.657 | 0.606 | `C=0.01` |
| SVM lineal | 0.507 ± 0.030 | 0.655 | 0.599 | `C=0.01` |
| SVM RBF | 0.549 ± 0.011 | 0.664 | 0.622 | `C=0.1, gamma=scale` |
| Árbol de decisión | 0.509 ± 0.045 | 0.620 | 0.585 | `max_depth=None, min_leaf=1` |
| K-NN | 0.584 ± 0.043 | 0.571 | 0.600 | `euclidean, k=5, distance` |

SVM RBF gana con ICN de 0.622 y el std más bajo del lote (0.011), lo que indica resultados muy estables entre folds. K-NN tiene el F1 macro más alto pero balanced accuracy muy baja, otra vez señal de que ignora la clase minoritaria.

### 10.7 Resumen: mejor modelo por target

| Target   | Mejor modelo (ICN) | F1 macro | Comentario |
|---|---|---:|---|
| GDS      | SVM RBF | 0.335 | 7 clases, k=2, caso muy difícil |
| GDS_R1   | SVM RBF | 0.671 | Empate técnico con K-NN |
| GDS_R2   | SVM RBF | 0.675 | SVM RBF domina |
| GDS_R3   | SVM RBF | 0.792 | Empate con K-NN en F1 |
| GDS_R4   | Regresión Logística | 0.532 | Único donde LR gana |
| GDS_R5   | SVM RBF | 0.549 | SVM RBF muy estable |

**SVM RBF gana en 5 de 6 targets.** Esto valida la hipótesis de que el modelo no lineal es mejor cuando hay suficiente estructura para aprenderla, pero cuando el problema es difícil por escasez de datos, una Regresión Logística regularizada puede ser más competitiva.

---

## 11. Comparación SVM lineal vs SVM RBF

Para responder la pregunta sobre si el kernel RBF aporta:

| Target | SVM lineal F1 | SVM RBF F1 | Diferencia | RBF gana? | McNemar-Yates significativo? |
|---|---:|---:|---:|:---:|:---:|
| GDS    | 0.308 | 0.335 | +0.027 | Sí | Sí (p<0.001) |
| GDS_R1 | 0.654 | 0.671 | +0.017 | Sí | No (p≈0.32) |
| GDS_R2 | 0.667 | 0.675 | +0.008 | Sí | No (p≈0.40) |
| GDS_R3 | 0.757 | 0.792 | +0.034 | Sí | Sí (p<0.001) |
| GDS_R4 | 0.521 | 0.550 | +0.029 | Sí | Sí (p<0.05) |
| GDS_R5 | 0.507 | 0.549 | +0.042 | Sí | Sí (p<0.001) |

**RBF gana en los 6 targets por F1 macro.** La diferencia es estadísticamente significativa en 4 de 6 targets según McNemar-Yates. En GDS_R1 y GDS_R2 la mejora es modesta y no es significativa. En problemas linealmente separables o con pocas muestras, un SVM lineal puede ser suficiente.

---

## 12. Sesgo de selección de hiperparámetros (Δsesgo)

El Δsesgo mide la diferencia entre el F1 macro reportado por `GridSearchCV` (optimista, sobre el set interno) y el F1 macro real (sobre el fold externo):

```text
Δsesgo = F1_macro_interno - F1_macro_externo
```

Δsesgo positivo = optimismo. Valores típicos: 0.005 a 0.05 en este laboratorio.

| Target | LR | SVM lin | SVM RBF | Árbol | K-NN |
|---|---:|---:|---:|---:|---:|
| GDS    | 0.013 | 0.015 | -0.012 | 0.080 | 0.107 |
| GDS_R1 | 0.010 | -0.017 | 0.013 | 0.032 | 0.037 |
| GDS_R2 | 0.001 | 0.006 | -0.001 | -0.004 | 0.012 |
| GDS_R3 | 0.010 | 0.007 | 0.005 | 0.021 | 0.014 |
| GDS_R4 | 0.005 | 0.000 | 0.018 | 0.017 | 0.008 |
| GDS_R5 | 0.015 | 0.011 | -0.008 | 0.006 | -0.001 |

Observaciones:

- El **Árbol de Decisión** muestra el mayor Δsesgo en `GDS` (0.080), lo que confirma que sobreajusta a las pocas muestras de las clases raras.
- **K-NN** también muestra Δsesgo alto en `GDS` (0.107) por la misma razón.
- Varios modelos dan Δsesgo **negativo** (RBF en GDS, GDS_R1, GDS_R2, GDS_R5). Esto se interpreta como ruido muestral: el fold externo resultó más fácil que el interno por azar, no como "validación más conservadora". Con k=2 o k=5 y datos pequeños, esto ocurre con frecuencia.

---

## 13. Tests de significancia (Wilcoxon + McNemar)

El laboratorio aplica dos tests pareados por target sobre los 6 modelos (15 pares por target, 90 totales).

### Wilcoxon firmado pareado

Compara las distribuciones de F1 macro fold a fold. Con solo k_outer ∈ {2, 5} folds, el test tiene muy poca potencia.

| Target | Pares con p<0.05 |
|---|---:|
| Todos | 0 |

Ninguna comparación dio p<0.05 con Wilcoxon. Esto es esperable: con k=5 folds pareados, Wilcoxon detecta solo diferencias grandes (>0.10 en F1). En `GDS` (k=2), la implementación además retorna p=1.0 cuando las diferencias son ~0 entre folds, lo cual es correcto pero refuerza que Wilcoxon no es informativo con tan pocos folds. Se incluye en el reporte por transparencia, no como criterio de decisión.

### McNemar con corrección de continuidad de Yates

Compara aciertos y fallos en los mismos ejemplos entre dos modelos. Yates se aplica porque b+c < 25 es común con datasets pequeños.

| Target | Pares significativos | % |
|---|---:|---:|
| GDS | 13/15 | 87% |
| GDS_R1 | 11/15 | 73% |
| GDS_R2 | 6/15 | 40% |
| GDS_R3 | 14/15 | 93% |
| GDS_R4 | 14/15 | 93% |
| GDS_R5 | 10/15 | 67% |

McNemar-Yates es más sensible que Wilcoxon. Detecta diferencias en casi todos los pares de GDS_R3, GDS_R4 y GDS_R5, pero menos en GDS_R2 (donde los modelos rinden parecido). El detalle está en `outputs/tables/significance_tests.csv` y los heatmaps en `outputs/figures/analysis/significance_heatmap_*.png`.

---

## 14. Bootstrap CI al 95%

Para cada modelo y target se calcularon intervalos de confianza al 95% por bootstrap (percentiles) sobre F1 macro, balanced accuracy y recall macro, con 1000 remuestras (`outputs/tables/bootstrap_ci_95.csv`).

Hallazgos principales:

- Los IC para F1 macro suelen tener ancho de 0.05 a 0.10, lo que refleja la inestabilidad real de los modelos en datasets pequeños.
- En `GDS_R3` (binario, n_min=172), el F1 macro de SVM RBF es 0.792 con IC95 = [0.763, 0.824]. Este es el resultado más estable del laboratorio.
- En `GDS`, los IC son muy anchos porque k_outer=2, lo que da solo 2 puntos para hacer bootstrap.

---

## 15. Clases con recall muy bajo

`outputs/tables/zero_recall_classes.csv` lista las clases con recall = 0.0:

| Target | Modelo | Clase | Support |
|---|---|---:|---:|
| GDS   | Árbol de decisión | 6 | 20 |
| GDS   | K-NN              | 7 | 2  |

Adicionalmente, `outputs/tables/low_support_classes.csv` lista todas las clases con soporte menor a 10. En `GDS`, la clase 6 (20 muestras) y la clase 7 (2 muestras) son estructuralmente difíciles para todos los modelos.

No se reportan clases con recall=0 fuera de `GDS`. El resto de los targets tiene `n_min >= 22`, suficiente para que los modelos aprendan al menos un patrón.

---

## 16. Árboles de decisión visualizados

`outputs/figures/analysis/decision_tree_{target}.png` muestra el árbol ajustado en el primer fold externo para cada target.

Observaciones:

- En `GDS_R3` (binario), el árbol es muy poco profundo (`max_depth=2` seleccionado por GridSearchCV). Solo usa 1-2 atributos.
- En `GDS`, el árbol necesita mucha más profundidad para separar 7 clases, lo que explica su bajo desempeño.
- Los atributos de **orientación temporal** (`Día`, `Mes`, `Año`, `Estación`) aparecen consistentemente en la raíz de los árboles, confirmando que son los más informativos.

---

## 17. Feature importance

`outputs/figures/analysis/feature_importance_{target}.png` muestra la importancia de cada feature por target y por modelo, comparando:

- Coeficientes absolutos de **Regresión Logística**
- Coeficientes absolutos de **SVM lineal**
- `feature_importances_` del **Árbol de decisión**

SVM RBF no expone `coef_` directamente (mapea a un espacio de dimensión infinita), por lo que no aparece en el heatmap.

La importancia reportada es el **promedio** de los `coef_` / `feature_importances_` de los `best_estimator_` de los 5 folds externos, lo que da una estimación más estable que ajustar un solo modelo sobre todos los datos.

Hallazgo principal: en todos los targets, los atributos más importantes son los de **orientación temporal** y **orientación espacial**. Los atributos de memoria verbal y geográfica aportan mucho menos.

---

## 18. Estabilidad de hiperparámetros

`outputs/figures/analysis/hyperparam_stability_{target}.png` muestra la frecuencia con que cada combinación de hiperparámetros ganó en los folds externos.

Observaciones:

- Para LR y SVM lineal, la grilla colapsa fuertemente en `C=0.01` en casi todos los targets. Esto sugiere que la regularización alta es consistentemente mejor.
- Para SVM RBF, la mejor config varía: en `GDS_R3` es `C=0.1, gamma=0.01` (5/5 folds), pero en `GDS_R4` es `C=0.1, gamma=1` (3/5 folds), mostrando que la elección de `gamma` es crítica y sensible al target.
- Para K-NN, no hay una configuración dominante. En algunos targets gana `k=11 uniform`, en otros `k=3 distance`. Esto explica su mayor varianza.

---

## 19. Comparación con feature engineering y agrupación de clases raras

Adicionalmente, se corrieron dos mejoras sobre el pipeline base para `GDS`:

1. **Feature engineering (FE)**: agregar 21 features de interacción AND entre pares de atributos del mismo grupo semántico (orientación temporal, orientación espacial, memoria verbal, memoria geográfica).
2. **Agrupación de clases raras (GROUP)**: colapsar las clases de `GDS` con menos de 50 muestras en una sola clase "severo" (nuevo label = max_label + 1). Esto convierte las 7 clases en 4.

Resultados sobre `GDS` (F1 macro por modelo):

| Modelo | BASE | FE | GROUP | FE+GROUP |
|---|---:|---:|---:|---:|
| Regresión Logística | 0.295 | 0.287 | 0.446 | 0.450 |
| SVM lineal | 0.308 | 0.294 | 0.412 | 0.440 |
| SVM RBF | 0.335 | 0.319 | 0.460 | 0.464 |
| Árbol de decisión | 0.223 | 0.268 | 0.404 | 0.407 |
| K-NN | 0.317 | 0.329 | 0.452 | 0.482 |

Conclusiones:

- **FE por sí solo no ayuda**: agregar 21 interacciones AND sobre 15 features introduce mucho ruido y los modelos no mejoran. En el árbol incluso mejora un poco, pero en el resto empeora.
- **GROUP sí ayuda mucho**: pasar de 7 clases a 4 (agrupando las 2 clases más raras) sube el F1 macro entre 0.10 y 0.18 en todos los modelos. La razón es que `n_min` sube de 2 a 64, lo que permite usar k_outer=5 en lugar de 2.
- **FE+GROUP** es marginalmente mejor que GROUP solo, pero la mejora no compensa la complejidad agregada.

`outputs/figures/comparisons/f1_comparison_baseline_vs_improvements.png` muestra esta comparación visualmente.

**Decisión final:** se mantiene el pipeline BASE con `k_outer=2` y se declara la limitación en el informe, en lugar de agrupar clases artificialmente. La razón es que agrupar clases cambia la semántica del problema y oculta el hecho de que `GDS` con 7 clases es estructuralmente difícil.

---

## 20. Curvas de aprendizaje

`outputs/figures/analysis/learning_curves.png` muestra la curva de F1 macro vs tamaño de entrenamiento para los 5 modelos sobre los 6 targets.

Hallazgo principal: las curvas se estabilizan muy rápido, lo que confirma que **más datos no resolverían el problema fundamental** del desbalance severo. La excepción es `GDS_R3` (binario), donde las curvas siguen subiendo suavemente, indicando que más datos sí ayudarían marginalmente.

---

## 21. Limitaciones

El proyecto tiene las siguientes limitaciones:

- El dataset tiene 1119 observaciones y un desbalance severo, especialmente en `GDS` (clase 7 con 2 muestras).
- El número de folds externos es bajo (2 para `GDS`, 5 para el resto), lo que reduce la potencia de los tests estadísticos.
- Las grillas de hiperparámetros son reducidas a propósito, lo que puede dejar fuera configuraciones ganadoras. Se eligió así para reducir el riesgo de selección optimista.
- El `class_weight="balanced"` es una solución simple al desbalance. Técnicas más elaboradas (SMOTE, undersampling) podrían explorarse en trabajos futuros, pero introducen complejidad adicional que el laboratorio no pide.
- Los modelos son clásicos (scikit-learn). No se usaron redes neuronales, ni ensembles como Random Forest o Gradient Boosting.
- El Δsesgo en el árbol para `GDS` (0.080) muestra que el modelo sobreajusta; los resultados para clases muy raras deben interpretarse como evidencia exploratoria, no concluyente.

---

## 22. Configuración final seleccionada

La configuración final del laboratorio fue:

```text
modelos       = Dummy + LR + SVM lineal + SVM RBF + Árbol + K-NN
validación    = anidada con k_outer = min(5, n_min), k_inner = min(3, k_outer)
grillas       = reducidas (≤ 16 combinaciones por modelo)
métrica       = f1_macro para selección interna
features      = 15 atributos binarios (sin feature engineering)
targets       = GDS, GDS_R1, GDS_R2, GDS_R3, GDS_R4, GDS_R5
reproducibilidad = seed outer=42, seed inner=123
```

Comando final recomendado:

```bash
python src/main.py --keep-estimators
```

Para reproducir las comparaciones de mejoras:

```bash
python src/compare_improvements.py
python src/plot_comparisons.py
```

---

## 23. Conclusiones

El laboratorio logró implementar un flujo completo de Machine Learning para clasificación de niveles de deterioro cognitivo usando cinco clasificadores fundamentales sobre seis formulaciones distintas del mismo problema.

Se partió con una línea base que solo tenía Regresión Logística y se fueron agregando los demás modelos, EDA, análisis avanzado, plots, tests de significancia, bootstrap, y comparación de mejoras incluyendo un baseline trivial y variantes de feature engineering y agrupación.

**Hallazgos principales:**

1. **SVM RBF es el mejor modelo en 5 de 6 targets.** La diferencia con SVM lineal es estadísticamente significativa en 4 de 6 targets (McNemar-Yates), pero la mejora en F1 macro es modesta (0.01 a 0.04).
2. **El target más difícil es `GDS`** con 7 clases y razón de desbalance 250:1. Todos los modelos quedan por debajo de 0.34 de F1 macro. La causa no es el modelo, es el dataset.
3. **`GDS_R3` (binario) es el más fácil** y el más estable, con F1 macro ~0.79 y std ~0.03.
4. **Regresión Logística es competitiva** en targets con estructura clara, e incluso gana en `GDS_R4` por su estabilidad.
5. **El Árbol de Decisión es el peor en casi todos los targets** y tiene el mayor Δsesgo, lo que confirma que sobreajusta a las clases raras.
6. **K-NN es sensible a la elección de k y la distancia**, sin una configuración dominante entre folds. Gana en F1 macro en 3 targets pero con balanced accuracy baja, lo que indica que ignora la clase minoritaria.
7. **El baseline DummyClassifier queda muy por debajo** en todos los casos (F1 entre 0.14 y 0.50), lo que confirma que los modelos están aprendiendo patrones reales.
8. **Agrupar clases raras ayuda mucho** (sube F1 macro entre 0.10 y 0.18 en `GDS`), pero cambia la semántica del problema, por lo que no se recomienda como solución final.
9. **El feature engineering con interacciones AND no aporta**: con 15 features y 1119 muestras, agregar 21 interacciones introduce más ruido que información.

**Recomendaciones para el informe:**

- Reportar siempre F1 macro, balanced accuracy, recall macro y matriz de confusión.
- Discutir explícitamente las clases con soporte muy bajo (clase 6 y 7 de `GDS`).
- Justificar el uso de `class_weight="balanced"` y de `k_outer = min(5, n_min)`.
- Mencionar que los resultados para `GDS` deben interpretarse como evidencia exploratoria.

La reproducibilidad está garantizada con semillas fijas (outer=42, inner=123) y la ejecución completa toma ~45 segundos en un servidor con `n_jobs=-1`.
