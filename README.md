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
│   │   ├── eda/
│   │   ├── analysis/
│   │   ├── experiments/
│   │   └── comparisons/
│   ├── confusion_matrices/
│   ├── per_class/
│   ├── estimator_cache/
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

Se implementaron los cinco clasificadores fundamentales pedidos por el enunciado, cada uno dentro de un `Pipeline` con `StandardScaler` (excepto el árbol). Adicionalmente se incluye un sexto modelo baseline para referencia:

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

El laboratorio usa **validación cruzada anidada** para separar correctamente la selección de hiperparámetros de la evaluación final. La implementación es funcionalmente equivalente a `cross_validate(estimator=GridSearchCV(...), cv=outer_cv, ...)`, como sugiere el enunciado, pero con un bucle manual que permite persistir métricas, estimadores y advertencias por fold:

```text
Ciclo externo (k_outer) -> estima el desempeño final
Ciclo interno (k_inner) -> selecciona hiperparámetros con GridSearchCV
```

El conjunto de prueba externo nunca se usa para elegir hiperparámetros ni para reportar resultados parciales.

### Número de folds

El número de folds se adapta al soporte de la clase menos representada, siguiendo el criterio del enunciado:

```text
k_outer = min(5, n_min)
k_inner = max(2, min(3, k_outer))
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

Para ejecutar el flujo principal (EDA + 30 experimentos de los 5 clasificadores sobre 6 objetivos, más 6 baselines dummy, + análisis + plots):

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

| Modelo | F1 macro | Balanced Acc. | ICN crudo | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.140 | 0.139 | 0.182 | default |
| Regresión Logística | 0.295 ± 0.038 | 0.440 | 0.395 | `C=0.01` |
| SVM lineal | 0.308 ± 0.023 | 0.470 | 0.415 | `C=0.01` |
| SVM RBF | 0.335 ± 0.039 | 0.452 | 0.419 | `C=10, gamma=0.1` |
| Árbol de decisión | 0.223 ± 0.033 | 0.412 | 0.348 | `max_depth=5, min_leaf=5` |
| K-NN | 0.317 ± 0.033 | 0.327 | 0.357 | `manhattan, n_neighbors=3, uniform` |

SVM RBF gana en ICN. Sin embargo, todos los modelos están por debajo de 0.34 de F1 macro, lo que muestra que `GDS` con 7 clases y 2 muestras en la clase más rara es un problema muy difícil. El baseline (0.140) confirma que los modelos sí están aprendiendo por encima de predecir según la distribución marginal.

**Nota:** por su k_outer=2, este caso se reporta como descriptivo. Los tests de significancia no son informativos con tan pocos folds.

### 10.2 GDS_R1 (3 clases, k=5)

| Modelo | F1 macro | Balanced Acc. | ICN crudo | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.337 | 0.334 | 0.368 | default |
| Regresión Logística | 0.651 ± 0.063 | 0.765 | 0.712 | `C=0.01` |
| SVM lineal | 0.654 ± 0.056 | 0.770 | 0.716 | `C=0.01` |
| SVM RBF | 0.671 ± 0.058 | 0.781 | 0.729 | `C=0.1, gamma=0.01` |
| Árbol de decisión | 0.592 ± 0.051 | 0.710 | 0.662 | `max_depth=3, min_leaf=1` |
| K-NN | 0.689 ± 0.064 | 0.676 | 0.700 | `euclidean, n_neighbors=11, distance` |

K-NN tiene el F1 macro más alto (0.689), pero SVM RBF gana en ICN (0.729) y en balanced accuracy (0.781). La diferencia entre ambos no es estadísticamente significativa según McNemar-Yates.

### 10.3 GDS_R2 (3 clases, k=5, n_min=172)

| Modelo | F1 macro | Balanced Acc. | ICN crudo | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.312 | 0.312 | 0.345 | default |
| Regresión Logística | 0.663 ± 0.043 | 0.666 | 0.679 | `C=0.01` |
| SVM lineal | 0.667 ± 0.048 | 0.662 | 0.681 | `C=0.01` |
| SVM RBF | 0.675 ± 0.056 | 0.667 | 0.687 | `C=0.1, gamma=0.01` |
| Árbol de decisión | 0.648 ± 0.032 | 0.658 | 0.669 | `max_depth=4, min_leaf=1` |
| K-NN | 0.635 ± 0.034 | 0.618 | 0.648 | `euclidean, n_neighbors=7, uniform` |

Este es el target con el desbalance más suave (ratio 3.8:1). SVM RBF lidera por ICN crudo (0.687), pero la diferencia con SVM lineal (0.681) es mínima. McNemar-Yates no detecta diferencias significativas en este target. Por ICN normalizado, la Regresión Logística gana en este target gracias a su mayor estabilidad entre folds.

### 10.4 GDS_R3 (binario, k=5)

| Modelo | F1 macro | Balanced Acc. | ICN crudo | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.499 | 0.499 | 0.522 | default |
| Regresión Logística | 0.764 ± 0.031 | 0.842 | 0.806 | `C=0.01` |
| SVM lineal | 0.757 ± 0.043 | 0.836 | 0.800 | `C=0.01` |
| SVM RBF | 0.792 ± 0.039 | 0.838 | 0.818 | `C=0.1, gamma=0.01` |
| Árbol de decisión | 0.733 ± 0.077 | 0.795 | 0.771 | `max_depth=2, min_leaf=1` |
| K-NN | 0.791 ± 0.033 | 0.764 | 0.792 | `euclidean, n_neighbors=11, uniform` |

GDS_R3 es binario y el problema más "fácil" del laboratorio. SVM RBF y K-NN tienen F1 macro casi idéntico (0.792 vs 0.791), y SVM RBF gana en ICN. El baseline (0.499) muestra que el problema es casi adivinar al ser binario y balanceado, pero los modelos reales ganan por +0.30.

### 10.5 GDS_R4 (3 clases, k=5, n_min=64)

| Modelo | F1 macro | Balanced Acc. | ICN crudo | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.302 | 0.301 | 0.335 | default |
| Regresión Logística | 0.532 ± 0.015 | 0.718 | 0.637 | `C=0.01` |
| SVM lineal | 0.521 ± 0.025 | 0.709 | 0.629 | `C=0.01` |
| SVM RBF | 0.550 ± 0.039 | 0.590 | 0.591 | `C=0.1, gamma=1` |
| Árbol de decisión | 0.515 ± 0.027 | 0.659 | 0.603 | `max_depth=None, min_leaf=1` |
| K-NN | 0.601 ± 0.042 | 0.593 | 0.619 | `euclidean, n_neighbors=3, uniform` |

Este es uno de los dos targets donde **Regresión Logística gana en ICN normalizado**. K-NN tiene el F1 macro más alto (0.601), pero su balanced accuracy (0.593) es la segunda más baja del lote, lo que indica que ignora parcialmente la clase minoritaria. SVM RBF tiene la balanced accuracy más baja (0.590) por un margen mínimo, aunque gana en F1 entre los modelos "complejos". Regresión Logística con `class_weight="balanced"` obtiene F1=0.532 y balanced accuracy=0.718, el balance más alto del lote.

### 10.6 GDS_R5 (3 clases, k=5, n_min=149)

| Modelo | F1 macro | Balanced Acc. | ICN crudo | Mejor config |
|---|---:|---:|---:|---|
| Baseline (Dummy) | 0.327 | 0.325 | 0.360 | default |
| Regresión Logística | 0.521 ± 0.029 | 0.657 | 0.606 | `C=0.01` |
| SVM lineal | 0.507 ± 0.030 | 0.655 | 0.599 | `C=0.01` |
| SVM RBF | 0.549 ± 0.011 | 0.664 | 0.622 | `C=0.1, gamma=scale` |
| Árbol de decisión | 0.509 ± 0.045 | 0.620 | 0.585 | `max_depth=None, min_leaf=1` |
| K-NN | 0.584 ± 0.043 | 0.571 | 0.600 | `euclidean, n_neighbors=5, distance` |

En GDS_R5, SVM RBF gana por ICN (0.622) con el std más bajo del lote (0.011), lo que indica resultados muy estables entre folds. K-NN tiene el F1 macro más alto (0.584 vs 0.549 de RBF) pero su balanced accuracy es la peor del lote (0.571), otra vez señal de que ignora la clase minoritaria. Es el mismo patrón de GDS_R1 y GDS_R3: K-NN optimiza la métrica de acierto global pero pierde recall en las clases con menos soporte.

### 10.7 Resumen: mejor modelo por target

Para ordenar modelos **dentro de un mismo target** se usa el ICN normalizado (compara modelos en la misma escala de dificultad). Para comparar **entre targets** se usa el ICN crudo, que conserva las magnitudes absolutas de las métricas. La tabla 10.7a muestra el ganador por ICN normalizado (el criterio de selección de cada target), mientras que 10.7b muestra el mejor ICN crudo y la posición de cada target en el ranking global de dificultad.

**Tabla 10.7a — Mejor modelo por target según ICN normalizado:**

El criterio de selección dentro de cada target es el **ICN normalizado** (escala 0–1, comparable solo entre modelos del mismo target). La columna *ICN crudo* muestra el valor absoluto del ganador.

| Target   | Mejor modelo          | F1 macro | ICN crudo | Comentario                                          |
|----------|-----------------------|---------:|----------:|-----------------------------------------------------|
| GDS      | SVM RBF               |    0.335 |     0.419 | 7 clases, k=2, caso muy difícil                     |
| GDS_R1   | SVM RBF               |    0.671 |     0.729 | Empate técnico con K-NN por F1 (0.689), RBF gana por BA |
| GDS_R2   | Regresión Logística   |    0.663 |     0.679 | LR gana por ICN_norm; SVM RBF lidera por ICN crudo y F1 |
| GDS_R3   | SVM RBF               |    0.792 |     0.818 | Diferencia mínima con K-NN en F1 (0.792 vs 0.791), RBF gana por ICN |
| GDS_R4   | Regresión Logística   |    0.532 |     0.637 | LR gana por estabilidad y BA; K-NN lidera por F1    |
| GDS_R5   | SVM RBF               |    0.549 |     0.622 | SVM RBF con std=0.011, el más estable del lote     |

**SVM RBF gana en 4 de 6 targets por ICN**, Regresión Logística gana en 2 (\texttt{GDS\_R2} y \texttt{GDS\_R4}). Por F1 macro, la distribución es distinta: SVM RBF gana en 3 targets (GDS, GDS_R2, GDS_R3) y K-NN en 3 (GDS_R1, GDS_R4, GDS_R5). Esto refleja que K-NN optimiza la métrica global pero pierde balanced accuracy, mientras que SVM RBF y LR tienen mejor equilibrio entre F1 y BA. La métrica más adecuada para ordenar modelos en un problema desbalanceado es el ICN, que pondera F1, BA, recall, precision y estabilidad.

**Tabla 10.7b — Ranking de dificultad de los targets (ICN crudo del mejor modelo, ordenado de mayor a menor):**

| Rank | Target   | Mejor F1 | Mejor ICN crudo | n_clases | n_min | k_outer |
|-----:|----------|---------:|----------------:|---------:|------:|--------:|
|    1 | GDS_R3   |    0.792 |           0.818 |        2 |   172 |       5 |
|    2 | GDS_R1   |    0.671 |           0.729 |        3 |    22 |       5 |
|    3 | GDS_R2   |    0.675 |           0.687 |        3 |   172 |       5 |
|    4 | GDS_R4   |    0.532 |           0.637 |        3 |    64 |       5 |
|    5 | GDS_R5   |    0.549 |           0.622 |        3 |   149 |       5 |
|    6 | GDS      |    0.335 |           0.419 |        7 |     2 |       2 |

**Lectura del ranking:** la dificultad del problema no depende solo del número de clases. GDS_R4 (3 clases, n_min=64) y GDS_R5 (3 clases, n_min=149) son más difíciles que GDS_R1 (3 clases, n_min=22), porque las clases intermedias de GDS_R4 y GDS_R5 están más solapadas en el espacio de features. Lo que finalmente pesa es la **separabilidad** de las clases en el espacio de los 15 atributos, no la cantidad de clases ni el desbalance por sí solos. GDS es intrínsecamente el más difícil porque sus 7 niveles colapsan en el mismo conjunto binario de respuestas correctas/incorrectas, lo que hace que las clases 5, 6 y 7 sean casi indistinguibles.

---

## 11. Comparación SVM lineal vs SVM RBF

Esta es la comparación más controlada del laboratorio, porque ambos modelos comparten la misma familia (SVM), la misma grilla para `C` y el mismo `class_weight="balanced"`. La única diferencia es si el espacio de features se mapea a uno de mayor dimensión vía kernel RBF o no. Si RBF gana consistentemente, hay evidencia de que el problema tiene estructura no lineal.

| Target   | SVM lineal F1 | SVM RBF F1 | ΔF1 (RBF - lin) | RBF gana? | McNemar-Yates p-value | Significativo? |
|----------|--------------:|-----------:|----------------:|:---------:|----------------------:|:--------------:|
| GDS      |         0.308 |      0.335 |           +0.028 |    Sí     |                 0.808 |       No       |
| GDS_R1   |         0.654 |      0.671 |           +0.017 |    Sí     |                 0.200 |       No       |
| GDS_R2   |         0.667 |      0.675 |           +0.008 |    Sí     |                 0.341 |       No       |
| GDS_R3   |         0.757 |      0.792 |           +0.034 |    Sí     |              < 0.001 |       Sí       |
| GDS_R4   |         0.521 |      0.550 |           +0.029 |    Sí     |              < 0.001 |       Sí       |
| GDS_R5   |         0.507 |      0.549 |           +0.042 |    Sí     |              < 0.001 |       Sí       |

**Lectura:** SVM RBF gana en los 6 targets por F1 macro, pero la magnitud de la mejora varía entre +0.008 (GDS_R2) y +0.042 (GDS_R5). La diferencia es estadísticamente significativa solo en 3 de 6 targets según McNemar-Yates (GDS_R3, GDS_R4, GDS_R5). En GDS, GDS_R1 y GDS_R2 la mejora no alcanza significancia, lo que indica que en estos targets el problema es aproximadamente separable por una frontera lineal y RBF no aporta valor real más allá de la complejidad. En GDS el test no tiene potencia por $k_{\text{outer}}=2$, por lo que la diferencia observada (+0.028) no se puede confirmar estadísticamente.

Un detalle relevante: en GDS_R4, RBF gana por F1 macro pero pierde por ICN, porque su balanced accuracy es 0.590 contra 0.718 de LR. Esto es coherente con un patrón típico de sobreajuste local: el kernel RBF ajusta muy bien las observaciones de entrenamiento de la clase mayoritaria, pero pierde recall en las clases minoritarias. En un dataset con solo 64 muestras en la clase minoritaria, la frontera no lineal tiene más riesgo de memorizar que de generalizar.

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
- Varios modelos dan Δsesgo **negativo** (RBF en GDS, GDS_R2, GDS_R5; SVM lineal en GDS_R1 y GDS_R4; Árbol en GDS_R2; K-NN en GDS_R5). Esto se interpreta como ruido muestral: el fold externo resultó más fácil que el interno por azar, no como "validación más conservadora". Con k=2 o k=5 y datos pequeños, esto ocurre con frecuencia.

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

| Target   | Modelo            | Clase | Support |
|----------|-------------------|------:|--------:|
| GDS      | Baseline (Dummy)  |     6 |      20 |
| GDS      | Baseline (Dummy)  |     7 |       2 |
| GDS      | Árbol de decisión |     6 |      20 |
| GDS      | K-NN              |     7 |       2 |
| GDS_R1   | Baseline (Dummy)  |     3 |      22 |

Adicionalmente, `outputs/tables/low_support_classes.csv` lista las clases con soporte menor a 10. En `GDS`, la clase 7 (2 muestras) es la única con soporte menor a 10; la clase 6 (20 muestras) está sobre el umbral, pero igual aparece como clase estructuralmente difícil por su solapamiento con la 5 y la 4 (ver matriz de confusión de la sección 15.1).

El `Baseline (Dummy)` no es un modelo entrenado: cuando una clase cae en folds donde su soporte de test es 0 (caso típico de clases raras) o cuando el clasificador marginal no la predice, su recall es 0. Esto es esperable y no es un defecto del pipeline. Fuera de las filas del Dummy, no se reportan clases con recall=0 en el resto de los targets.

### 15.1 Análisis de matrices de confusión agregadas

El recall por clase del mejor modelo de cada target (SVM RBF en 4/6 casos, Regresión Logística en GDS_R2 y GDS_R4) muestra dónde fallan los clasificadores y revela patrones distintos según la formulación del problema. Las matrices se construyen concatenando las predicciones de los 5 folds externos (o 2 en GDS), de modo que cada celda representa el total de ejemplos del set de test.

| Target   | Clase mayoritaria recall | Clase minoritaria recall | Confusión típica |
|----------|-------------------------:|-------------------------:|------------------|
| GDS      | clase 1 → 0.58 (n=149)  | clase 7 → 1.00 (n=2)     | clases 1↔2, 2↔3 adyacentes se confunden |
| GDS_R1   | clase 1 → 0.83 (n=947)  | clase 3 → 0.82 (n=22)    | clase 1 ↔ 2 (154 errores)                |
| GDS_R2   | clase 1 → 0.79 (n=649)  | clase 2 → 0.54 (n=298)   | clase 1 → 2 (131) y 2 → 1 (89)           |
| GDS_R3   | clase 1 → 0.89 (n=947)  | clase 3 → 0.79 (n=172)   | errores balanceados entre las dos clases |
| GDS_R4   | clase 2 → 0.53 (n=906)  | clase 1 → 0.77 (n=149)   | clase 2 → 1 (345), clase 2 → 3 (82)      |
| GDS_R5   | clase 2 → 0.50 (n=798)  | clase 3 → 0.79 (n=172)   | clase 2 → 3 (118), clase 1 → 2 (41)     |

**Lectura por target:**

- **GDS** (7 clases, k=2): la matriz muestra la estructura típica de un problema ordinal mal aprendido. Los errores se concentran entre clases adyacentes en la escala (1↔2, 2↔3, 3↔4), que es el patrón esperado cuando los modelos lineales o RBF no tienen suficiente resolución ordinal. La clase 7 muestra recall=1.0 con solo 2 muestras, pero esto es un artefacto: al ser k=2 con la clase 7 entera en un solo fold, ambos ejemplos cayeron en el mismo fold de test y se predijeron correctamente. **No se debe interpretar como "el modelo aprendió la clase 7"**, sino como ruido muestral.
- **GDS_R1** (3 clases): la clase mayoritaria (1) tiene 947 muestras, pero aun así se confunde 154 veces con la clase 2. La clase minoritaria (3, n=22) tiene un recall sorprendentemente alto (0.82), lo que confirma que `class_weight="balanced"` está funcionando.
- **GDS_R2** (3 clases, el más balanceado): la clase 1 (n=649) se confunde con la 2 (131 errores directos), la clase 2 (n=298) envía 89 errores a la 1 y 49 a la 3, y la clase 3 (n=172) envía 48 errores a la 2. Los errores están en la frontera ordinal, lo que sugiere que la escala subyacente es continua y los modelos la aproximan con clases discretas. El mejor modelo aquí es Regresión Logística (no SVM RBF) por ICN, con F1 macro 0.663.
- **GDS_R3** (binario, el más fácil): los errores están bien balanceados (104 de la clase 1 van a 3, 37 de la 3 van a 1). El modelo predice la clase mayoritaria con recall=0.89.
- **GDS_R4** (3 clases, n_min=64): la mejor matriz es de Regresión Logística (no SVM RBF). La clase mayoritaria (2, n=906) tiene recall=0.53, lo que indica que se confunde mucho con la clase 1 (345 errores van de 2 a 1) y con la clase 3 (82 errores van de 2 a 3). Esto explica por qué LR gana en este target: con frontera aproximadamente lineal y estructura de clases difusa, RBF sobreajusta localmente mientras que LR generaliza mejor. SVM RBF tiene mayor F1 macro (0.550) pero menor balanced accuracy (0.590 vs 0.718 de LR).
- **GDS_R5** (3 clases): la clase 2 (n=798) tiene recall de solo 0.50, y la clase 1 (n=149) se va 41 veces a la 2. La distribución de errores muestra que el target tiene solapamiento real entre las clases 1 y 2, no un problema del modelo.

**Conclusión general sobre las matrices:** los errores se concentran en clases adyacentes en la escala ordinal, lo que es la firma de un problema donde la escala GDS es intrínsecamente continua y los modelos la discretizan. Esto refuerza la idea de que el problema no se resuelve con mejores modelos, sino con datos más finos o con formulaciones que exploten la naturaleza ordinal (regresión ordinal, por ejemplo).

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

### 17.1 Importancia agregada por dimensión cognitiva

Para entender qué dimensión cognitiva aporta más a la predicción de cada nivel de deterioro, se agruparon los 15 atributos en las 4 dimensiones del test y se sumó la importancia normalizada de LR (los coeficientes absolutos del SVM lineal son prácticamente idénticos). Los valores son la suma de las importancias normalizadas dentro de cada dimensión, por lo que suman 1.0 en cada fila:

| Target   | Orient. temporal | Orient. espacial | Memoria verbal | Memoria geográfica | Feature top |
|----------|-----------------:|-----------------:|---------------:|-------------------:|-------------|
| GDS      | 0.267 | 0.358 | 0.159 | 0.216 | `País`      |
| GDS_R1   | 0.298 | 0.333 | 0.147 | 0.222 | `Ciudad`    |
| GDS_R2   | 0.307 | 0.117 | 0.227 | 0.349 | `A682`      |
| GDS_R3   | 0.291 | 0.118 | 0.192 | 0.398 | `Copiapo2`  |
| GDS_R4   | 0.395 | 0.148 | 0.254 | 0.203 | `Mes`       |
| GDS_R5   | 0.280 | 0.113 | 0.247 | 0.360 | `Copiapo2`  |

**Lectura:** la importancia de las dimensiones cambia según el target. Para `GDS` y `GDS_R1` (que conservan la granularidad alta de la escala), predominan **orientación temporal** y **espacial** — son las dimensiones más sensibles al deterioro cognitivo temprano, porque un fallo en `País` o `Ciudad` ya indica desorientación. En cambio, para los targets reagrupados en 3 clases (`GDS_R2`, `GDS_R5`) y para el binario (`GDS_R3`), la **memoria geográfica** gana peso (top feature: `A682` o `Copiapo2`). Esto tiene sentido clínico: cuando el deterioro es moderado a severo, las fallas se generalizan y la memoria se vuelve más discriminante que la orientación. El top feature varía entre los 6 targets (País, Ciudad, A682, Copiapo2, Mes, Copiapo2), lo que confirma que no hay un único atributo dominante, sino que la formulación del problema cambia qué feature es más informativa.

---

## 18. Estabilidad de hiperparámetros

`outputs/figures/analysis/hyperparam_stability_{target}.png` muestra la frecuencia con que cada combinación de hiperparámetros ganó en los folds externos.

Observaciones:

- Para LR y SVM lineal, la grilla colapsa fuertemente en `C=0.01` en casi todos los targets. Esto sugiere que la regularización alta es consistentemente mejor.
- Para SVM RBF, la mejor config varía: en `GDS_R3` es `C=0.1, gamma=0.01` (5/5 folds), pero en `GDS_R4` es `C=0.1, gamma=1` (3/5 folds), mostrando que la elección de `gamma` es crítica y sensible al target.
- Para K-NN, no hay una configuración dominante. En algunos targets gana `k=11 uniform`, en otros `k=3 distance`. Esto explica su mayor varianza.

---

## 19. Análisis adicional: feature engineering y agrupación de clases raras

> **Nota:** esta sección corresponde a un análisis exploratorio **adicional** al requerimiento del laboratorio. El enunciado pide los 5 clasificadores sobre los 6 objetivos originales (15 atributos binarios). Los experimentos de interacciones y agrupación se ejecutaron solo sobre `GDS` para explorar si la dificultad extrema de ese target podría mitigarse.

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

## 19.1 Costo computacional e interpretabilidad

Más allá de la métrica, el PDF pide comparar modelos por **equilibrio entre desempeño, interpretabilidad y estabilidad**. La tabla resume el costo de entrenamiento por fold externo (medido sobre `GDS_R2`, n=895 entrenamiento, n=224 test) y el tipo de explicación que cada modelo ofrece:

| Modelo              | Fit time (s) | Tipo de explicación                            | Hiperparámetros interpretables |
|---------------------|-------------:|------------------------------------------------|--------------------------------|
| Regresión Logística |        0.006 | Coeficientes por feature y por clase          | `C` (regularización)            |
| SVM lineal          |        0.016 | Coeficientes del hiperplano                    | `C` (ancho de margen)          |
| Árbol de decisión   |        0.001 | Reglas `if-then` con soporte y confianza      | `max_depth`, `min_samples_leaf`|
| K-NN                |        0.003 | Prototipos del vecindario (caso por caso)      | `k`, `metric`, `weights`       |
| SVM RBF             |        0.025 | No interpretable directamente (kernel implícito)| `C`, `gamma` (ancho de banda) |

**Lectura:** todos los modelos son baratos de entrenar con 1119 observaciones, así que el costo no fue un criterio de selección. La diferencia relevante está en la **interpretabilidad**:

- **Árbol de decisión** es el único que entrega reglas explícitas del tipo "si `Mes=1` y `Año=1` entonces clase 1". En el laboratorio, los árboles resultantes tienen profundidad 2-5 (sección 16), lo que los hace directamente traducibles a una guía clínica: por ejemplo, "si falla orientación temporal pero conserva memoria, está en GDS_R2; si falla ambas, está en GDS_R3".
- **Regresión Logística y SVM lineal** entregan coeficientes, que son interpretables pero requieren normalización. Los signos de los coeficientes (sección 17) confirman que acertar `Mes`, `Año` y `NúmeroPiso` empuja la predicción hacia clases leves, y fallarlos hacia clases severas.
- **SVM RBF** mapea a un espacio de dimensión infinita, por lo que no es directamente interpretable. Es un modelo de "caja negra" cuyo valor está en el desempeño, no en la explicación.
- **K-NN** no tiene parámetros aprendidos: cada predicción es un voto de los k vecinos más cercanos. Es localmente interpretable ("¿qué casos se parecen a este?") pero globalmente no entrega una regla.

**Implicación para el informe:** si el uso final es clínico, el Árbol y la Regresión Logística son los más defendibles, porque su predicción puede ser auditada. Si el uso es de cribado (screening) donde solo importa detectar deterioro, SVM RBF es la mejor opción por desempeño. K-NN queda en un punto intermedio: tiene buen F1 pero su comportamiento depende fuertemente de la métrica de distancia, lo que lo hace menos confiable para producción.

## 19.2 Efecto de `class_weight="balanced"`

Todos los modelos que lo soportan (LR, SVM lineal, SVM RBF, Árbol) usan `class_weight="balanced"`, que asigna un peso inversamente proporcional a la frecuencia de cada clase en la función de pérdida. K-NN no lo soporta; en la grilla explorada, `weights="uniform"` ganó en 21 folds externos y `weights="distance"` en 6, por lo que la mayoría de los modelos K-NN ajustados usan `uniform`. Esta elección se justificó en el diseño porque sin `class_weight`, los modelos tenderían a predecir siempre la clase mayoritaria, elevando la accuracy por encima del 80% pero colapsando el F1 macro bajo 0.10 en los targets desbalanceados.

Una validación indirecta del efecto se observa en GDS_R1: la clase minoritaria (3, n=22) tiene recall=0.82 con `class_weight="balanced"`, mientras que el baseline DummyClassifier tiene recall=0 sobre esa misma clase. Sin el balanceo, un modelo ingenuo predeciría siempre clase 1 (n=947) y lograría accuracy≈0.85 con F1 macro≈0.16. Con el balanceo, F1 macro sube a 0.65-0.69.

No se corrió un experimento controlado con y sin `class_weight` porque aumentaría la matriz experimental 6×6×2 = 72 corridas, lo que el laboratorio no pidió y el criterio de "grilla reducida" desaconseja para evitar más riesgo de selección optimista.

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
validación    = anidada con k_outer = min(5, n_min), k_inner = max(2, min(3, k_outer))
grillas       = reducidas (4-20 combinaciones por modelo)
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

El laboratorio permitió comparar cinco clasificadores fundamentales sobre seis formulaciones distintas del mismo problema de deterioro cognitivo. A continuación se integra lo aprendido, organizando la discusión por pregunta del enunciado en lugar de por modelo.

### ¿Cuál clasificador gana en cada target?

SVM RBF gana en 3 targets por F1 macro (GDS, GDS_R2, GDS_R3) y K-NN en 3 (GDS_R1, GDS_R4, GDS_R5). Por ICN normalizado, SVM RBF gana en 4 de 6 y Regresión Logística en 2 (GDS_R2 y GDS_R4). Esta diferencia es importante: K-NN optimiza la métrica global pero pierde balanced accuracy, mientras que SVM RBF y LR tienen mejor equilibrio entre F1 y BA. La métrica más adecuada para ordenar modelos en un problema desbalanceado es el ICN, que pondera F1, BA, recall, precision y estabilidad. Los casos donde LR gana (GDS_R2 y GDS_R4) son la confirmación práctica del mensaje central del laboratorio: el modelo más complejo no siempre gana.

### ¿La Regresión Logística es competitiva frente a modelos no lineales?

Sí. En `GDS_R2` y `GDS_R4` gana por ICN gracias a su buena balanced accuracy (0.666 y 0.718 vs 0.667 y 0.590 de RBF) y, en el caso de GDS_R4, su baja varianza entre folds (std=0.015). En los demás targets queda dentro de 0.04 de F1 del mejor modelo. Su grilla colapsa en `C=0.01` consistentemente (3-5 de 5 folds externos, dependiendo del target), lo que sugiere que la regularización alta es lo que le da estabilidad, no el modelo en sí. La diferencia es que SVM RBF ajusta mejor el sesgo de los datos de entrenamiento, pero a costa de mayor varianza entre folds.

### ¿El kernel RBF mejora realmente al SVM lineal?

Sí, pero con matices. Gana en los 6 targets por F1 macro con mejoras entre +0.008 (GDS_R2) y +0.042 (GDS_R5). La mejora es estadísticamente significativa solo en 3 de 6 targets según McNemar-Yates (GDS_R3, GDS_R4, GDS_R5). En GDS, GDS_R1 y GDS_R2, donde la estructura del problema es aproximadamente lineal o los folds son muy pocos, la mejora no es significativa y la complejidad de RBF no se justifica estadísticamente. En GDS el test no tiene potencia por $k_{\text{outer}}=2$.

### ¿El Árbol de decisión entrega reglas interpretables sin perder demasiado desempeño?

Sí en interpretabilidad, no en desempeño. Los árboles resultantes tienen profundidad 2-5 (sección 16), con 1-3 reglas del tipo "si `Mes=1` y `Año=1` entonces clase 1", directamente traducibles a una guía clínica. En desempeño, el Árbol es el peor en 4 de 6 targets por F1 macro, y su Δsesgo es el más alto en GDS (0.080), lo que confirma que sobreajusta a las clases raras. La conclusión es que vale la pena mostrar el árbol como herramienta explicativa, pero no como modelo competitivo.

### ¿K-NN es estable o depende demasiado de k y la distancia?

Depende. K-NN gana en F1 macro en 3 targets (GDS_R1, GDS_R3, GDS_R5), pero en todos ellos su balanced accuracy es la peor del lote, lo que indica que ignora la clase minoritaria. La configuración ganadora varía entre folds sin un patrón claro (n_neighbors=3 a n_neighbors=11, euclidean/manhattan, uniform/distance), lo que lo hace el modelo menos confiable para producción. Para el informe, vale la pena mostrar que K-NN tiene buen F1 pero no es robusto al desbalance, a diferencia de los SVM con `class_weight="balanced"`.

### ¿Qué objetivo es el más difícil de predecir y por qué?

`GDS` es el más difícil (F1≈0.34, ICN crudo 0.419) por una combinación de 7 clases muy desbalanceadas y `k_outer=2`, lo que da una evaluación casi descriptiva. Pero la observación más interesante del ranking (tabla 10.7b) es que `GDS_R5` (3 clases, n_min=149) es más difícil que `GDS_R1` (3 clases, n_min=22), porque las tres clases de GDS_R5 están más solapadas. La dificultad no depende del número de clases ni del desbalance por sí solos, sino de la **separabilidad** de las clases en el espacio de features. Esto se confirma con las matrices de confusión (sección 15.1): los errores se concentran en clases adyacentes, lo que indica que la escala GDS es intrínsecamente continua y los modelos la discretizan.

### ¿Existen clases con recall muy bajo?

Sí, en `GDS` exclusivamente (considerando los cinco modelos entrenados, no el Baseline Dummy). La clase 6 (n=20) tiene recall=0.25 con SVM RBF y 0.0 con Árbol, y la clase 7 (n=2) tiene recall=0.0 con Árbol y K-NN. Estas son las dos clases estructuralmente más raras (sección 15). En los demás targets no hay clases con recall=0, lo que confirma que `n_min ≥ 22` es suficiente para que los modelos aprendan al menos un patrón.

### ¿Hay evidencia de sobreajuste en algún modelo?

Sí, principalmente en el Árbol y K-NN sobre `GDS`, con Δsesgo de 0.080 y 0.107 respectivamente (sección 12). Esto significa que el F1 macro reportado por `GridSearchCV` sobre el set interno era 0.08-0.10 más alto que el F1 real sobre el fold externo. El SVM RBF muestra Δsesgo negativo en varios targets, lo que se interpreta como ruido muestral (el fold externo resultó más fácil por azar), no como subajuste. La regularización explícita de LR y SVM lineal (con `C=0.01`) parece controlar el sobreajuste de manera efectiva.

### ¿Qué decisión metodológica influyó más en los resultados?

Tres decisiones explican la mayor parte de la varianza observada:

1. **`k_outer = min(5, n_min)`** adapta el número de folds al soporte de la clase minoritaria, evitando que un fold externo quede sin representación de alguna clase. Sin esta regla, el fold externo en `GDS_R1` con `k=10` habría tenido folds con 0-1 ejemplos de la clase 3, y la evaluación sería inválida.
2. **`class_weight="balanced"`** evita que los modelos predigan siempre la clase mayoritaria. Sin esto, un SVM lineal sobre `GDS_R1` predeciría siempre clase 1, accuracy 0.85, F1 macro 0.16. Con esto, F1 macro sube a 0.65-0.69.
3. **Grillas reducidas (≤20 combinaciones por modelo)** limitan el riesgo de selección optimista de hiperparámetros. Esto es particularmente importante en un dataset donde k=5 da solo 5 puntos para evaluar el fold externo, y donde una grilla más grande permitiría "ganar" por azar.

### Trabajo futuro

Tres líneas de continuación son naturales para extender este laboratorio:

1. **Modelos ordinales**: usar regresión logística ordinal o `mord` para explotar el orden natural de la escala GDS, en lugar de tratarla como clasificación nominal.
2. **Técnicas de remuestreo para desbalance**: probar SMOTE, undersampling o `class_weight` por clase, en lugar del balanceo global, para ver si la clase 6 o 7 de GDS se pueden rescatar.
3. **Validación con datos externos**: el modelo se ajustó y evaluó sobre el mismo dataset; una validación con datos de otro centro clínico sería el siguiente paso para confirmar la generalización.

### Recomendaciones para el informe

- **Reportar siempre F1 macro, balanced accuracy, recall macro y matriz de confusión.** Accuracy es engañosa en problemas desbalanceados.
- **Discutir explícitamente las clases con soporte muy bajo** (clase 6 y 7 de GDS) y declarar que sus resultados son evidencia exploratoria.
- **Justificar el uso de `class_weight="balanced"` y de `k_outer = min(5, n_min)`** en la sección de metodología.
- **Mostrar al menos una matriz de confusión por target** (sección 15.1 ya las tiene listas) para que el lector vea dónde fallan los modelos.
- **Distinguir entre "ganador por F1" y "ganador por ICN"** cuando los rankings no coincidan (caso GDS_R4), porque el criterio de selección cambia la conclusión.

La reproducibilidad está garantizada con semillas fijas (outer=42, inner=123) y la ejecución completa toma ~45 segundos en un servidor con `n_jobs=-1`.
