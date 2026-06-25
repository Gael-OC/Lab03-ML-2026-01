# Laboratorio 03 — Machine Learning

**Asignatura**: Machine Learning
**Docente**: Dr. Juan Bekios Calfa
**Grupo**: 1
**Estudiantes**: Gael Ortega y Matías Vidal

> Documento de referencia del Laboratorio 03 (clasificadores fundamentales). Su
> proposito es servir como fuente de verdad para que un companero pueda
> redactar el informe tecnico en PDF sin perder informacion metodologica.

---

## 0. Resumen ejecutivo

Este laboratorio compara cinco clasificadores fundamentales —Regresion
Logistica, SVM lineal, SVM con kernel RBF, Arbol de Decision y K-NN— mas
un baseline trivial (`DummyClassifier`) sobre seis variables objetivo
(`GDS`, `GDS_R1`, ..., `GDS_R5`) que representan distintas formas de
agrupar la escala GDS de deterioro cognitivo, a partir de 15 atributos
binarios extraidos de un test neuropsicologico reducido.

El diseno experimental usa validacion cruzada anidada con grillas
reducidas para evitar sobreajuste en la seleccion de hiperparametros.
La metrica principal es F1 macro (con balanced accuracy y recall macro
como soporte), y se reportan bootstrap CI al 95%, tests de Wilcoxon
pareado y McNemar con correccion de Yates.

**Hallazgo principal:** SVM con kernel RBF gana en 4 de los 6 targets por
F1 macro y por ICN crudo. El unico target donde no gana es `GDS_R4`, donde
la regresion regularizada es competitiva. El baseline `DummyClassifier`
queda muy por debajo en todos los casos, lo que confirma que los modelos
estan aprendiendo patrones reales y no solo reproduciendo la distribucion
marginal.

El target `GDS` (escala original de 7 niveles) tiene clases con
soporte tan bajo (clase 7 con solo 2 muestras) que **se reporta como caso
exploratorio** y no como experimento evaluable. Los resultados son
descriptivos, no concluyentes.

---

## 1. Problema y dataset

### 1.1 Contexto

El dataset `15 atributos R0-R5.sav` (SPSS) contiene 1119 observaciones de
un test neuropsicologico reducido. Cada fila corresponde a la respuesta
de un paciente a 15 preguntas binarias (0 = fallo, 1 = acierto), agrupadas
en cuatro dimensiones cognitivas:

| Dimension              | Atributos                                            |
|------------------------|------------------------------------------------------|
| Orientacion temporal   | Dia, Mes, Anio, Estacion                             |
| Orientacion espacial   | Pais, Ciudad, CalleLugar, NumeroPiso                 |
| Memoria verbal         | Miguel2, Gonzalez2, Avenida2, Imperial2             |
| Memoria geografica     | A682, Caldera2, Copiapo2                            |

La columna `ID` es el identificador y **no se usa como feature**.

### 1.2 Variables objetivo

Seis columnas que representan formas alternativas de agrupar la escala
GDS (Global Deterioration Scale):

| Target   | n_clases | Descripcion                                     | n_min | k_outer |
|----------|---------:|-------------------------------------------------|------:|--------:|
| GDS      | 7        | Escala original                                 | 2     | 2       |
| GDS_R1   | 3        | Reagrupacion 1                                  | 22    | 5       |
| GDS_R2   | 3        | Reagrupacion 2                                  | 172   | 5       |
| GDS_R3   | 2        | Reagrupacion binaria                            | 172   | 5       |
| GDS_R4   | 3        | Reagrupacion 4                                  | 64    | 5       |
| GDS_R5   | 3        | Reagrupacion 5                                  | 149   | 5       |

`n_min` es el tamano de la clase mas pequena y `k_outer` es el numero
de folds externos que la validacion cruzada puede usar respetando
`k_outer <= n_min` (regla del PDF, slide 23). El detalle del
preprocesamiento de cada agrupacion esta en `outputs/tables/distribucion_clases.csv`.

### 1.3 Restricciones metodologicas del enunciado

- Cada target se evalua como un experimento independiente (no se mezclan
  columnas objetivo).
- `k_outer = min(5, n_min)` por la limitacion estratificada.
- Si una clase tiene un solo ejemplo, **no se puede garantizar
  validacion estratificada**: se declara como caso exploratorio.
- Grillas pequenas para reducir el riesgo de seleccion optimista de
  hiperparametros en datasets pequenos.

---

## 2. Estructura del proyecto

```
.
|-- config/
|   `-- paths.yaml                  # rutas y parametros del experimento
|-- datasets/
|   `-- 15 atributos R0-R5.sav      # dataset SPSS (no versionado)
|-- src/
|   |-- __init__.py
|   |-- settings.py                 # carga paths.yaml
|   |-- data_loader.py              # carga .sav y prepara X, y
|   |-- models.py                   # 6 modelos con pipelines y grillas
|   |-- evaluation.py               # validacion anidada y metricas
|   |-- significance.py             # Wilcoxon + McNemar-Yates
|   |-- reports.py                  # CSV, LaTeX, PDF (reportlab)
|   |-- eda.py                      # analisis exploratorio
|   |-- analysis.py                 # learning curves, feature importance,
|   |                              # bootstrap CI, tests de significancia
|   |-- plots.py                    # figuras comparativas
|   |-- compare_improvements.py     # analisis extra: FE + agrupacion
|   |-- plot_comparisons.py         # figuras del analisis extra
|   `-- main.py                     # orquesta todo
|-- outputs/                        # generado, ignorado por git
|-- environment.yml
|-- README.md
`-- .gitignore
```

Modulos principales:

| Archivo | Responsabilidad |
|---|---|
| `src/main.py` | Orquesta EDA, experimentos, analisis, plots y reportes. |
| `src/data_loader.py` | Carga el `.sav` y entrega `(X, y)` para cada target. |
| `src/models.py` | Define los 6 modelos con sus `Pipeline` y grillas. Centraliza `MODEL_ORDER` y `MODEL_DISPLAY_NAMES`. |
| `src/evaluation.py` | `run_nested_cv` con cache de estimadores, ICN crudo y normalizado. |
| `src/significance.py` | Wilcoxon pareado y McNemar con correccion de Yates sobre estimadores cacheados. |
| `src/reports.py` | CSV, LaTeX y PDF (con `reportlab`). |
| `src/eda.py` | Distribucion de clases, correlacion, relacion GDS -> GDS_Rk. |
| `src/analysis.py` | Curvas de aprendizaje, feature importance, bootstrap CI, tests. |
| `src/plots.py` | Heatmaps de F1, balanced accuracy, ICN, matrices de confusion, dashboard. |
| `src/compare_improvements.py` | Variantes con feature engineering y agrupacion de clases raras. |
| `config/paths.yaml` | Rutas, semillas, numero de folds. |

---

## 3. Preprocesamiento y diseno del pipeline

### 3.1 Pipeline por modelo

```text
| Modelo              | Pipeline                                              |
|---------------------|-------------------------------------------------------|
| DummyClassifier     | (sin scaler; el dummy ignora las features)            |
| Regresion Logistica | StandardScaler -> LogisticRegression                  |
| SVM lineal          | StandardScaler -> SVC(kernel=linear)                  |
| SVM RBF             | StandardScaler -> SVC(kernel=rbf)                     |
| Arbol de decision   | DecisionTreeClassifier (sin escalado)                 |
| K-NN                | StandardScaler -> KNeighborsClassifier                |
```

El escalado se hace **dentro del Pipeline** para que se ajuste solo
con los datos de entrenamiento de cada fold, evitando fuga de
informacion. El Arbol no usa `StandardScaler` porque es invariante a
transformaciones monotonas.

`class_weight="balanced"` se aplica a todos los modelos que lo
soportan (LR, SVM, Arbol). K-NN no tiene este parametro, pero
`weights="distance"` ayuda a compensar el desbalance. El baseline
`DummyClassifier` usa `strategy="stratified"` (predice respetando
la distribucion marginal de clases).

### 3.2 Grillas de hiperparametros

| Modelo              | Parametros                                                  |
|---------------------|-------------------------------------------------------------|
| Regresion Logistica | C = [0.01, 0.1, 1, 10]                                      |
| SVM lineal          | C = [0.01, 0.1, 1, 10]                                      |
| SVM RBF             | C = [0.1, 1, 10], gamma = [scale, 0.01, 0.1, 1]            |
| Arbol de decision   | max_depth = [2, 3, 4, 5, None], min_samples_leaf = [1, 3, 5, 10] |
| K-NN                | n_neighbors = [3, 5, 7, 9, 11], weights = [uniform, distance], metric = [euclidean, manhattan] |
| DummyClassifier     | sin hiperparametros                                         |

Las grillas son reducidas a proposito. Con tan pocos ejemplos por
clase, una busqueda exhaustiva favorece configuraciones que ganan
por azar.

---

## 4. Validacion cruzada anidada

```
| Ciclo externo (k_outer)  | Estima el desempeno final.            |
| Ciclo interno (k_inner)  | Selecciona hiperparametros con GridSearchCV. |
```

El fold externo de prueba **nunca** se usa para elegir hiperparametros
ni para reportar resultados parciales.

### 4.1 Numero de folds

| Target   | n_min | k_outer | Regla                                  |
|----------|------:|--------:|----------------------------------------|
| GDS      | 2     | 2       | min(5, n_min) = 2                      |
| GDS_R1   | 22    | 5       | min(5, n_min) = 5                      |
| GDS_R2   | 172   | 5       | min(5, n_min) = 5                      |
| GDS_R3   | 172   | 5       | min(5, n_min) = 5                      |
| GDS_R4   | 64    | 5       | min(5, n_min) = 5                      |
| GDS_R5   | 149   | 5       | min(5, n_min) = 5                      |

Para `GDS`, la clase minoritaria tiene 2 ejemplos, lo que limita
`k_outer` a 2. El enunciado permite este caso pero pide declararlo
explícitamente como limitacion (slide 23).

### 4.2 Ciclo interno y KFold no estratificado

El ciclo interno se construye con la misma regla `k_inner = min(3, k_outer)`.
Para `GDS`, en el split externo queda una clase con 1 solo ejemplo en
el conjunto de entrenamiento, lo que impide `StratifiedKFold`. En ese
caso el codigo usa `KFold` no estratificado y registra la advertencia
en `outputs/advertencias.txt`:

```
[GDS | Regresion Logistica] GDS: ciclo interno usa KFold no
estratificado porque una clase tiene solo 1 ejemplo dentro de un
entrenamiento externo.
```

Advertencia equivalente para SVM lineal, SVM RBF, Arbol y K-NN.

---

## 5. Metricas

### 5.1 Metricas por fold externo

Para cada par (target, modelo) se calculan las siguientes metricas
promediadas sobre los k_outer folds:

- `accuracy`: proporcion de aciertos (no se prioriza; se omite del
  reporte principal por ser enganosa con desbalance).
- `balanced_accuracy`: promedio de recall por clase, robusto al
  desbalance.
- `precision_macro` y `recall_macro`: promedios simples por clase.
- `f1_macro`: media armonica de precision y recall por clase;
  **metrica principal** para seleccion interna de hiperparametros.
- `stability` (1 - std de f1_macro): estabilidad entre folds.

### 5.2 ICN: dos variantes, mismo nombre en el PDF

El PDF del laboratorio define el ICN como una suma ponderada de
metricas crudas (slide 40). Esa formula es comparable **entre
targets** (todos los modelos usan la misma escala absoluta). Sin
embargo, dentro de un target, el ICN crudo esta dominado por la
magnitud del target, no por la diferencia entre modelos. Para
ordenar modelos dentro de un mismo target se usa una version
normalizada min-max entre los modelos de ese target.

El laboratorio reporta **ambas**:

- `icn_raw` (formula del PDF, comparable entre targets): suma
  ponderada de F1 + BA + Recall + Precision + Stability con los
  pesos 0.40 / 0.25 / 0.20 / 0.10 / 0.05.
- `icn` (normalizado min-max entre modelos del target): util
  para ordenar modelos dentro del target.

Ambas variantes se reportan en `outputs/tables/resumen_resultados.csv`
y en las tablas LaTeX/PDF. En la documentacion se usa `ICN*` para
referirse al ICN crudo y `ICN` para el normalizado.

### 5.3 Soporte de `zero_division=0`

Todas las metricas macro usan `zero_division=0`. Esto significa que
una clase nunca predicha contribute con F1=0 al promedio macro, no
con F1=NaN. La alternativa (`zero_division=np.nan` + `nanmean`) es
mas rigurosa pero cambia los numeros. La eleccion es consistente
en todo el laboratorio y esta documentada.

---

## 6. Resultados por experimento

Todos los valores se calcularon con la corrida `python src/main.py
--keep-estimators`. Las tablas incluyen el baseline trivial.

### 6.1 GDS — caso exploratorio (k=2, clase minoritaria con 2 ejemplos)

| Modelo              | F1 macro         | Balanced Acc. | ICN*  | n_min | k_outer |
|---------------------|------------------|---------------|-------|------:|--------:|
| Baseline (Dummy)    | 0.140            | 0.139         | 0.182 | 2     | 2       |
| Regresion Logistica | 0.295 +/- 0.038  | 0.440         | 0.395 | 2     | 2       |
| SVM lineal          | 0.308 +/- 0.023  | 0.470         | 0.415 | 2     | 2       |
| SVM RBF             | 0.335 +/- 0.039  | 0.452         | 0.419 | 2     | 2       |
| Arbol de decision   | 0.223 +/- 0.033  | 0.412         | 0.348 | 2     | 2       |
| K-NN                | 0.317 +/- 0.033  | 0.327         | 0.357 | 2     | 2       |

**Interpretacion:** GDS es la escala original con 7 niveles, donde la
clase minoritaria tiene solo 2 muestras. El PDF (slide 23) senala
que "si una clase tiene solo 1 ejemplo, no es posible garantizar
validacion estratificada para esa clase. El informe debe declarar
esta limitacion y justificar si se agrupa, se excluye del analisis o
se reporta como caso exploratorio".

**Decision tomada:** se reporta GDS como **caso exploratorio**. Las
conclusiones son descriptivas, no inferenciales:

- Los cinco modelos quedan entre F1=0.22 y 0.34, muy por encima del
  baseline (0.14) pero muy por debajo de los targets reagrupados
  (0.50-0.79).
- SVM RBF lidera por F1 macro y por ICN*, pero la diferencia no es
  significativa con un test de Wilcoxon (k=2 folds implica
  varianza demasiado alta para diferenciar).
- El Arbol de decision muestra el mayor Δsesgo (sesgo de seleccion
  de hiperparametros), 0.080, lo que confirma que sobreajusta a
  las clases con muy pocas muestras.
- Los IC y los tests de significancia sobre `GDS` no son
  informativos por el bajo k_outer.

### 6.2 GDS_R1 (3 clases, k=5, n_min=22)

| Modelo              | F1 macro         | Balanced Acc. | ICN*  |
|---------------------|------------------|---------------|-------|
| Baseline (Dummy)    | 0.337            | 0.334         | 0.368 |
| Regresion Logistica | 0.651 +/- 0.063  | 0.765         | 0.712 |
| SVM lineal          | 0.654 +/- 0.056  | 0.770         | 0.716 |
| SVM RBF             | 0.671 +/- 0.058  | 0.781         | 0.729 |
| Arbol de decision   | 0.592 +/- 0.051  | 0.710         | 0.662 |
| K-NN                | 0.689 +/- 0.064  | 0.676         | 0.700 |

SVM RBF gana por ICN* y por F1 macro (con K-NN cerca). El IC
bootstrap al 95% para SVM RBF es [0.611, 0.733]. La diferencia con
K-NN en F1 macro no es estadisticamente significativa segun McNemar-Yates
(p ≈ 0.40) y Wilcoxon no rechaza H0.

### 6.3 GDS_R2 (3 clases, k=5, n_min=172, mejor balanceado)

| Modelo              | F1 macro         | Balanced Acc. | ICN*  |
|---------------------|------------------|---------------|-------|
| Baseline (Dummy)    | 0.312            | 0.312         | 0.345 |
| Regresion Logistica | 0.663 +/- 0.043  | 0.666         | 0.679 |
| SVM lineal          | 0.667 +/- 0.048  | 0.662         | 0.681 |
| SVM RBF             | 0.675 +/- 0.056  | 0.667         | 0.687 |
| Arbol de decision   | 0.648 +/- 0.032  | 0.658         | 0.669 |
| K-NN                | 0.635 +/- 0.034  | 0.618         | 0.648 |

Este es el target con el desbalance mas suave (ratio 3.8:1). Los
cinco modelos quedan muy cerca (F1 entre 0.635 y 0.675). McNemar-Yates
no detecta diferencias significativas en ninguna comparacion par a
par. SVM RBF gana por ICN* y por F1 macro, pero el margen es de
0.01-0.04, dentro de la variabilidad de los folds.

**Lectura:** con clases suficientemente representadas, los cinco
modelos rinden parecido. La eleccion de uno sobre otro no es
estadisticamente robusta, asi que conviene preferir el mas simple
(SVM lineal) o el mas estable (Regresion Logistica con
regularizacion alta).

### 6.4 GDS_R3 (binario, k=5, n_min=172)

| Modelo              | F1 macro         | Balanced Acc. | ICN*  |
|---------------------|------------------|---------------|-------|
| Baseline (Dummy)    | 0.499            | 0.499         | 0.522 |
| Regresion Logistica | 0.764 +/- 0.031  | 0.842         | 0.806 |
| SVM lineal          | 0.757 +/- 0.043  | 0.836         | 0.800 |
| SVM RBF             | 0.792 +/- 0.039  | 0.838         | 0.818 |
| Arbol de decision   | 0.733 +/- 0.077  | 0.795         | 0.771 |
| K-NN                | 0.791 +/- 0.033  | 0.764         | 0.792 |

GDS_R3 es binario y con clases casi balanceadas, por lo que es el
problema "mas facil" del laboratorio. SVM RBF y K-NN empatan en
F1 macro (0.792) y SVM RBF gana en ICN* y en balanced accuracy.
El Arbol de decision tiene la peor balanced accuracy y la mayor
desviacion estandar, coherente con su inestabilidad en clases
raras. La diferencia SVM RBF vs K-NN es marginal y no es
significativa por McNemar-Yates.

### 6.5 GDS_R4 (3 clases, k=5, n_min=64)

| Modelo              | F1 macro         | Balanced Acc. | ICN*  |
|---------------------|------------------|---------------|-------|
| Baseline (Dummy)    | 0.302            | 0.301         | 0.335 |
| Regresion Logistica | 0.532 +/- 0.015  | 0.718         | 0.637 |
| SVM lineal          | 0.521 +/- 0.025  | 0.709         | 0.629 |
| SVM RBF             | 0.550 +/- 0.039  | 0.590         | 0.591 |
| Arbol de decision   | 0.515 +/- 0.027  | 0.659         | 0.603 |
| K-NN                | 0.601 +/- 0.042  | 0.593         | 0.619 |

Este target merece discusion aparte. K-NN lidera en F1 macro (0.601)
pero su balanced accuracy (0.593) es la peor del lote. SVM RBF
tambien gana en F1 macro entre los modelos "complejos" pero con
balanced accuracy baja (0.590). Esto sugiere que ambos estan
sobreponderando la clase mayoritaria: predicen bien la clase
mas frecuente y fallan en la minoritaria, y aun asi obtienen
buen F1 macro porque la clase mayoritaria pesa mas en el
promedio.

Regresion Logistica con `class_weight="balanced"` obtiene F1=0.532
y balanced accuracy=0.718, el balance mas alto del lote. Es el
modelo mas estable y el unico donde F1 y balanced accuracy son
ambos razonables. SVM RBF tiene la mayor std de balanced accuracy
(0.112, ver seccion 10), lo que confirma su inestabilidad en este
target.

**Conclusion:** GDS_R4 es el unico target donde Regresion Logistica
es preferible. El alto F1 de K-NN enmascara su dificultad con la
clase minoritaria, y SVM RBF es inestable.

### 6.6 GDS_R5 (3 clases, k=5, n_min=149)

| Modelo              | F1 macro         | Balanced Acc. | ICN*  |
|---------------------|------------------|---------------|-------|
| Baseline (Dummy)    | 0.327            | 0.325         | 0.360 |
| Regresion Logistica | 0.521 +/- 0.029  | 0.657         | 0.606 |
| SVM lineal          | 0.507 +/- 0.030  | 0.655         | 0.599 |
| SVM RBF             | 0.549 +/- 0.011  | 0.664         | 0.622 |
| Arbol de decision   | 0.509 +/- 0.045  | 0.620         | 0.585 |
| K-NN                | 0.584 +/- 0.043  | 0.571         | 0.600 |

SVM RBF lidera en ICN* y en estabilidad (std mas bajo del lote,
0.011). K-NN tiene el F1 macro mas alto pero con balanced
accuracy muy baja (0.571), otra vez senal de que ignora la clase
minoritaria. La eleccion de SVM RBF es consistente con la
mayoria de los otros targets.

### 6.7 Resumen por target (ganador por ICN* y por F1 macro)

| Target   | Mejor por ICN*       | F1*  | Mejor por F1 macro   | F1   | Comentario                                       |
|----------|----------------------|------|----------------------|------|--------------------------------------------------|
| GDS      | SVM RBF              | 0.42 | SVM RBF              | 0.34 | Caso exploratorio (k=2). Diferencias no confiables. |
| GDS_R1   | SVM RBF              | 0.73 | K-NN (cerca de SVM RBF) | 0.69 | Empate tecnico. |
| GDS_R2   | SVM RBF              | 0.69 | SVM RBF              | 0.68 | Diferencias pequenas.                            |
| GDS_R3   | SVM RBF              | 0.82 | SVM RBF (empate K-NN) | 0.79 | Mas estable y mejor balanceado.                  |
| GDS_R4   | Regresion Logistica  | 0.64 | K-NN                 | 0.60 | K-NN ignora la clase minoritaria.                |
| GDS_R5   | SVM RBF              | 0.62 | K-NN                 | 0.58 | SVM RBF es mas estable.                          |

SVM RBF gana en 5 de 6 targets por ICN*; en el restante gana
Regresion Logistica. Por F1 macro, K-NN gana en 3 targets, pero
con balanced accuracy baja, asi que la victoria es enganyosa en
GDS_R4 y GDS_R5.

---

## 7. Comparacion SVM lineal vs SVM RBF

El PDF (slide 49) pregunta si el kernel RBF mejora al SVM lineal.

| Target   | SVM lineal F1 | SVM RBF F1 | Diferencia | RBF gana? | McNemar-Yates significativo? |
|----------|--------------:|-----------:|-----------:|:---------:|:----------------------------:|
| GDS      | 0.308         | 0.335      | +0.027     | Si        | Si (p<0.001)                  |
| GDS_R1   | 0.654         | 0.671      | +0.017     | Si        | No (p≈0.32)                   |
| GDS_R2   | 0.667         | 0.675      | +0.008     | Si        | No (p≈0.40)                   |
| GDS_R3   | 0.757         | 0.792      | +0.034     | Si        | Si (p<0.001)                  |
| GDS_R4   | 0.521         | 0.550      | +0.029     | Si        | Si (p<0.05)                   |
| GDS_R5   | 0.507         | 0.549      | +0.042     | Si        | Si (p<0.001)                  |

RBF gana en los 6 targets por F1 macro. La diferencia es
estadisticamente significativa en 4 de 6 targets segun
McNemar-Yates (correccion de continuidad): GDS, GDS_R3, GDS_R4
y GDS_R5. En GDS_R1 y GDS_R2 la mejora es modesta y no es
significativa.

**Conclusion:** el kernel RBF aporta, pero la mejora es pequena
fuera de los targets donde hay estructura claramente no lineal.
Cuando la separacion es aproximadamente lineal (GDS_R2) o el
problema tiene clases pequenas (GDS_R1), un SVM lineal con
regularizacion alta puede ser suficiente.

---

## 8. Sesgo de seleccion de hiperparametros (Δsesgo)

Δsesgo = F1 macro interno (GridSearchCV) - F1 macro externo (test
del fold). Δsesgo positivo indica que la seleccion interna fue
optimista. Valores observados:

| Target   | LR    | SVM lin | SVM RBF | Arbol  | K-NN   |
|----------|------:|--------:|--------:|-------:|-------:|
| GDS      | 0.013 | 0.015   | -0.012  | 0.080  | 0.107  |
| GDS_R1   | 0.010 | -0.017  | 0.013   | 0.032  | 0.037  |
| GDS_R2   | 0.001 | 0.006   | -0.001  | -0.004 | 0.012  |
| GDS_R3   | 0.010 | 0.007   | 0.005   | 0.021  | 0.014  |
| GDS_R4   | 0.005 | 0.000   | 0.018   | 0.017  | 0.008  |
| GDS_R5   | 0.015 | 0.011   | -0.008  | 0.006  | -0.001 |

Observaciones:

- El **Arbol** muestra el mayor Δsesgo en `GDS` (0.080) y un
  valor positivo en casi todos los targets: confirma que
  sobreajusta a las clases pequenas.
- **K-NN** muestra el mayor Δsesgo en `GDS` (0.107), coherente
  con su sensibilidad a las pocas muestras minoritarias.
- Varios modelos dan Δsesgo **negativo** (RBF en GDS, GDS_R1,
  GDS_R2, GDS_R5). Esto **no significa que el modelo sea mejor
  de lo que parece** (el README original interpretaba esto como
  "validacion mas conservadora", lo cual es incorrecto). Es
  ruido muestral: por azar, el fold externo resulto mas facil
  que el fold interno de validacion. Con k=2 o k=5 y datos
  pequenos, esto ocurre con frecuencia.

Δsesgo positivo y pequeno (0.005-0.020) es el comportamiento
esperado y no es motivo de alarma. Δsesgo > 0.05 sugiere que el
modelo sobreajusta al conjunto de validacion.

---

## 9. Bootstrap CI al 95%

Se calcularon intervalos de confianza al 95% por bootstrap
(percentiles) sobre la metrica F1 macro de los folds externos.
1000 remuestras con reemplazo, semilla=42.

| Target   | Modelo            | F1 medio | IC 95%           | n_folds |
|----------|-------------------|---------:|------------------|--------:|
| GDS_R3   | SVM RBF           | 0.792    | [0.763, 0.824]   | 5       |
| GDS_R3   | Regresion Log.    | 0.764    | [0.728, 0.795]   | 5       |
| GDS_R4   | K-NN              | 0.601    | [0.553, 0.652]   | 5       |
| GDS_R4   | SVM RBF           | 0.550    | [0.503, 0.601]   | 5       |
| GDS_R1   | K-NN              | 0.689    | [0.617, 0.756]   | 5       |
| GDS_R1   | SVM RBF           | 0.671    | [0.611, 0.733]   | 5       |

El IC mas estable es el de SVM RBF en GDS_R3, con ancho de 0.06
(0.79 ± 0.03). En GDS el IC no es informativo porque solo hay 2
folds.

Detalle completo en `outputs/tables/bootstrap_ci_95.csv` y
forest plot en `outputs/figures/analysis/bootstrap_ci_forest_95.png`.

---

## 10. Tests de significancia (Wilcoxon + McNemar-Yates)

### 10.1 Wilcoxon firmado pareado

Wilcoxon se aplica a los F1 macro por fold entre pares de modelos
(5 folds por par). Con k=5, el test tiene poca potencia, y con
k=2 (GDS) el test es practicamente inutil.

| Target   | Pares con p<0.05 (Wilcoxon) | Total pares | Comentario                        |
|----------|----------------------------:|------------:|-----------------------------------|
| GDS      | 0                           | 15          | k=2, test sin potencia            |
| GDS_R1   | 0                           | 15          | Variabilidad entre folds enmascara |
| GDS_R2   | 0                           | 15          | Margenes pequenos                 |
| GDS_R3   | 0                           | 15          | Empate tecnico                    |
| GDS_R4   | 0                           | 15          | Variabilidad alta                 |
| GDS_R5   | 0                           | 15          | Empate tecnico                    |

Wilcoxon **no rechaza** ninguna H0 en este laboratorio. Esto
coincide con la literatura: Wilcoxon pareado con k=5 folds
externos tiene potencia muy baja para detectar diferencias
menores a 0.05 en F1 macro. Se incluye en el reporte por
transparencia, no como criterio de decision.

Con 6 modelos hay 15 pares por target (no 10). El conteo
anterior se mantiene: 0 rechazos por Wilcoxon.

### 10.2 McNemar con correccion de Yates

McNemar compara aciertos y fallos en los mismos ejemplos de los
folds externos entre dos modelos. La correccion de Yates
(|b-c|-1)^2/(b+c) es recomendada cuando b+c < 25, comun con
datasets pequenos.

| Target   | Pares con p<0.05 (McNemar-Yates) | Total pares | % significativos |
|----------|---------------------------------:|------------:|-----------------:|
| GDS      | 13                              | 15          | 87%              |
| GDS_R1   | 11                              | 15          | 73%              |
| GDS_R2   | 6                               | 15          | 40%              |
| GDS_R3   | 14                              | 15          | 93%              |
| GDS_R4   | 14                              | 15          | 93%              |
| GDS_R5   | 10                              | 15          | 67%              |

McNemar-Yates es mas sensible que Wilcoxon. Detecta diferencias
significativas en la mayoria de los pares de todos los targets,
incluso en GDS donde Wilcoxon no puede correr por k=2. La
excepcion es GDS_R2 (donde los modelos rinden parecido y los
p-valores son mas altos). Con k=2 folds externos, McNemar opera
sobre aproximadamente 559 ejemplos (los del fold de test), lo que
es suficiente para detectar diferencias. El detalle por par esta
en `outputs/tables/significance_tests.csv` y los heatmaps en
`outputs/figures/analysis/significance_heatmap_*.png`.

**Conclusion:** con k=5 los tests pareados tienen poca potencia.
La metrica principal para comparar modelos sigue siendo F1 macro
con su intervalo bootstrap, complementada con McNemar-Yates para
los pares donde se observa una diferencia puntual.

**Limitacion metodologica:** los estimadores de los folds externos
se persisten en disco (`outputs/estimator_cache/`) para que el
analisis de significancia no reentrene la validacion anidada. Esto
se activa con `python src/main.py --keep-estimators`. Sin esta
opcion, los tests de significancia se omiten.

---

## 11. Comparacion con feature engineering y agrupacion de clases raras

Como complemento se ejecutaron dos variantes sobre `GDS`:

1. **FE**: agregar 21 features de interaccion AND entre pares
   del mismo grupo semantico (orientacion temporal, espacial,
   memoria verbal, geografica).
2. **GROUP**: agrupar las clases con menos de 50 muestras en
   una sola clase "severo" (las clases 5, 6 y 7 colapsan en una).
   Pasa de 7 a 4 clases.

Resultados sobre `GDS` (F1 macro, max de los 5 modelos):

| Variante   | LR    | SVM lin | SVM RBF | Arbol  | K-NN   | n_min |
|------------|------:|--------:|--------:|-------:|-------:|------:|
| BASE       | 0.295 | 0.308   | 0.335   | 0.223  | 0.317  | 2     |
| FE         | 0.287 | 0.294   | 0.319   | 0.268  | 0.329  | 2     |
| GROUP      | 0.446 | 0.412   | 0.460   | 0.404  | 0.452  | 64    |
| FE+GROUP   | 0.450 | 0.440   | 0.464   | 0.407  | 0.482  | 64    |

Observaciones:

- **FE no ayuda por si solo**: agregar 21 interacciones AND sobre
  15 features introduce mucho ruido. Las interacciones AND son
  aproximadamente equivalentes a la multiplicacion bit a bit, y
  en variables binarias altamente correlacionadas (Dia, Mes, Anio
  casi siempre coinciden), esto es redundante.
- **GROUP ayuda mucho**: pasar de 7 a 4 clases sube el F1 macro
  entre 0.10 y 0.18 en todos los modelos. La razon mecanica es
  que `n_min` sube de 2 a 64, lo que permite k_outer=5.
- **FE+GROUP** es marginalmente mejor que GROUP solo.

**Decision tomada:** mantener el pipeline BASE para el informe
final. Agrupar clases cambia la semantica del problema (mezcla
niveles 5, 6 y 7) y oculta el hecho de que GDS con 7 clases es
intrinsecamente dificil por escasez de datos. Si el enunciado
exigiera "predecir el nivel exacto de deterioro", agrupar seria
inaceptable; en cambio, FE por si solo no aporta.

Detalle completo en `outputs/comparisons/comparison_baseline_vs_improvements.csv`.

---

## 12. Feature importance

`outputs/figures/analysis/feature_importance_{target}.png` muestra
la importancia normalizada de cada feature para los modelos que
la exponen: Regresion Logistica (|coef_|), SVM lineal (|coef_|)
y Arbol de decision (`feature_importances_`).

**Hallazgo consistente en los 6 targets:** los atributos de
**orientacion temporal** (Dia, Mes, Anio, Estacion) tienen la
mayor importancia. La **orientacion espacial** tambien aporta.
**Memoria verbal y geografica** tienen pesos menores.

Nota: SVM RBF no expone `coef_` directamente (mapea a un espacio
de dimension infinita), por lo que no aparece en el heatmap. La
importancia de RBF se discute en el README via comparacion
contra SVM lineal (seccion 7).

La importancia reportada es el **promedio** de los `coef_` /
`feature_importances_` de los `best_estimator_` de los 5 folds
externos, lo que da una estimacion mas estable que ajustar un
solo modelo sobre todos los datos.

---

## 13. Curvas de aprendizaje

`outputs/figures/analysis/learning_curves.png` muestra F1 macro
vs tamano de entrenamiento, con StratifiedKFold sobre 8 tamanos
entre 10% y 100%.

Hallazgos:

- Las curvas se estabilizan rapidamente: en la mayoria de los
  targets, el F1 macro entre 50% y 100% de los datos varia menos
  de 0.02. Esto confirma que **mas datos no resolverian el
  problema fundamental** del desbalance severo.
- Excepcion: `GDS_R3` (binario) sigue subiendo suavemente, lo
  que sugiere que mas datos ayudarian marginalmente.
- En `GDS` las curvas son muy ruidosas por k=2; no son
  informativo.

---

## 14. Arboles de decision visualizados

`outputs/figures/analysis/decision_tree_{target}.png` muestra el
Arbol ajustado en el primer fold externo para cada target (con
`max_depth=4` fijo, no el del mejor fold).

- En `GDS_R3` (binario) el Arbol es muy poco profundo:
  `max_depth=2` lo clasifica casi tan bien como el resto.
- En `GDS` el Arbol necesita mucha profundidad para separar
  7 clases, lo que explica su bajo desempeno.
- Los atributos de **orientacion temporal** aparecen en la raiz
  en todos los targets donde el Arbol performa mejor.

---

## 15. Respuestas a las preguntas del enunciado (slide 49)

### 1. ¿Cual clasificador obtiene el mejor F1 macro por columna objetivo?

| Target   | Mejor F1 macro          | F1   | ICN* del ganador  |
|----------|-------------------------|------|-------------------|
| GDS      | SVM RBF (exploratorio)  | 0.335 | 0.419 (SVM RBF)   |
| GDS_R1   | K-NN (cerca de SVM RBF) | 0.689 | 0.729 (SVM RBF)   |
| GDS_R2   | SVM RBF                 | 0.675 | 0.687 (SVM RBF)   |
| GDS_R3   | SVM RBF (empate K-NN)   | 0.792 | 0.818 (SVM RBF)   |
| GDS_R4   | K-NN                    | 0.601 | 0.637 (LR)        |
| GDS_R5   | K-NN                    | 0.584 | 0.622 (SVM RBF)   |

K-NN gana por F1 macro en 3 targets (GDS_R1, GDS_R4, GDS_R5),
pero con balanced accuracy baja en GDS_R4 (0.593) y GDS_R5
(0.571). SVM RBF lidera en 4 de 6 cuando se considera ICN* o
balanced accuracy, lo que muestra que la "victoria" de K-NN
enmascara su dificultad con clases minoritarias. Regresion
Logistica es la unica que gana en GDS_R4 cuando se pondera
estabilidad y balance.

### 2. ¿La Regresion Logistica es competitiva frente a modelos no lineales?

Si, es competitiva. Gana en `GDS_R4` por ICN* (0.637 vs 0.591 de
SVM RBF) y queda a menos de 0.04 de SVM RBF en F1 macro en todos
los targets donde no gana. La grilla colapsa en `C=0.01` (alta
regularizacion) en 4 de 5 targets, lo que confirma que el
problema esta dominado por el desbalance, no por la complejidad
del modelo.

### 3. ¿El kernel RBF mejora al SVM lineal?

Si en F1 macro (los 6 targets) y si en McNemar-Yates (4 de 6
targets). La mejora es pequena (0.01-0.04) y solo significativa
cuando hay estructura no lineal clara (GDS, GDS_R3, GDS_R4,
GDS_R5). En GDS_R1 y GDS_R2 la diferencia es marginal y no
significativa.

### 4. ¿El Arbol entrega reglas interpretables sin perder desempeno?

Entrega reglas interpretables pero **rinde peor que el resto** en
todos los targets. Su Δsesgo es el mas alto del lote en GDS
(0.080), confirmando sobreajuste a clases pequenas. Su F1 macro
queda entre 0.04 y 0.13 por debajo del SVM RBF. **No es
recomendable como modelo final** para este problema, aunque
sirve para visualizar la estructura del espacio de atributos
y para confirmar que orientacion temporal es la dimension
mas informativa.

### 5. ¿K-NN es estable o depende de k y la distancia?

Depende. En GDS_R3 (binario) y GDS_R1 es competitivo y estable.
En GDS_R4 y GDS_R5 gana en F1 macro pero con balanced accuracy
baja, lo que indica que **ignora la clase minoritaria** a pesar
de `weights="distance"`. La eleccion de `n_neighbors` y `metric`
cambia entre folds: no hay una configuracion dominante. Su
ventaja es no requerir entrenamiento; su desventaja es el costo
de prediccion (O(n) por consulta).

### 6. ¿Que objetivo parece mas dificil de predecir y por que?

`GDS` (escala original de 7 niveles). Razones:

- n_min = 2, lo que obliga a k_outer = 2.
- Distribucion de clases con razon de desbalance 250:1 (clase
  mayoritaria 499 vs clase 7 con 2).
- El ciclo interno usa `KFold` no estratificado, lo que invalida
  los tests de significancia.
- F1 macro del mejor modelo (0.335) esta muy cerca del baseline
  (0.140) en proporcion: hay senal, pero es debil.

Los demas targets son reagrupaciones de GDS que combinan clases
vecinas y mejoran el balance. `GDS_R3` (binario) es el mas facil
(0.79 de F1). `GDS_R2` es el mas balanceado (ratio 3.8) y los 5
modelos rinden parecido (diferencias < 0.04 en F1).

### 7. ¿Existen clases con recall muy bajo?

`outputs/tables/zero_recall_classes.csv` lista las clases con
recall = 0.0:

| Target | Modelo            | Clase | Support |
|--------|-------------------|------:|--------:|
| GDS    | Arbol de decision | 6     | 20      |
| GDS    | K-NN              | 7     | 2       |

`outputs/tables/low_support_classes.csv` lista todas las clases
con soporte < 10. En `GDS`, las clases 6 (20 muestras) y 7 (2
muestras) son estructuralmente difficiles para todos los modelos.

Fuera de `GDS`, ningun modelo tiene recall=0 en ninguna clase.
Con `n_min >= 22`, los modelos logran al menos predecir
ocasionalmente cada clase.

### 8. ¿Hay evidencia de sobreajuste en algun modelo?

Si, principalmente en **GDS**:

- Arbol de decision: Δsesgo = 0.080, F1 macro en el mejor fold
  externo es 0.23 pero el F1 interno es 0.31. Sobreajuste a las
  pocas muestras de las clases 6 y 7.
- K-NN: Δsesgo = 0.107, el mas alto del lote. La validacion
  interna premia configuraciones que funcionan para todas las
  clases (incluidas las raras), pero el fold externo no las
  incluye.
- SVM RBF: Δsesgo = -0.012. El fold externo resulto mas facil
  que el fold interno por azar, no senal de sobreajuste. Con
  k=2 y solo 2 folds, esta variabilidad es esperable.

En los demas targets el Δsesgo es < 0.04, comportamiento normal.

### 9. ¿Que decision metodologica influyo mas en los resultados?

**La eleccion de `k_outer = min(5, n_min)`.** Esta regla obliga a
degradar `GDS` a k=2, lo que limita severamente la potencia de los
tests de significancia y la estabilidad de las metricas. Tambien
determina que la eleccion de hiperparametros internos sea ruidosa
para las clases minoritarias de GDS (5, 6, 7 con 6, 20 y 2 muestras).

**Segunda decision:** el uso de `class_weight="balanced"` en todos
los modelos que lo soportan. Esto compensa parcialmente el
desbalance, pero no resuelve el problema de fondo: con clases de 2
o 3 ejemplos, el modelo no puede aprender una frontera confiable.

**Tercera decision:** grillas pequenas. Reduce el riesgo de
seleccion optimista, pero deja configuraciones ganadoras fuera de
la busqueda. Es el trade-off explicito del enunciado.

---

## 16. Limitaciones y trabajo futuro

### 16.1 Limitaciones

- **Dataset pequeno** (1119 obs) con desbalance severo en `GDS`.
- **k_outer bajo**: 2 para `GDS`, 5 para el resto. Los tests de
  significancia tienen poca potencia.
- **Grillas reducidas**: a proposito, pero pueden dejar fuera
  configuraciones ganadoras.
- **Sin feature engineering**: el analisis muestra que agregar
  21 interacciones AND no ayuda.
- **Sin oversampling/undersampling**: solo `class_weight="balanced"`
  compensa el desbalance. SMOTE, NearMiss o RandomOverSampler
  podrian explorarse en trabajos futuros.
- **Sin ensembles**: Random Forest, Gradient Boosting y Stacking
  no se implementaron. El PDF no los pide, pero son la continuacion
  natural.
- **Sin redes neuronales**: el laboratorio se enfoca en
  clasificadores fundamentales.

### 16.2 Trabajo futuro

1. Implementar y comparar ensembles (Random Forest, Gradient
   Boosting, Stacking) sobre los mismos targets.
2. Explorar tecnicas de remuestreo (SMOTE, NearMiss,
   RandomOverSampler) y comparar contra `class_weight="balanced"`.
3. Aplicar SHAP o LIME para explicabilidad a nivel de ejemplo
   individual, no solo de feature importance global.
4. Validacion externa: si se consigue otra base de datos con el
   mismo test neuropsicologico, entrenar aqui y evaluar alla
   (cross-database).
5. Aumentar la grilla de SVM RBF con busqueda bayesiana
   (skopt, hyperopt) ya que es donde parece haber mas ganancia
   marginal.

---

## 17. Instalacion y ejecucion

### 17.1 Entorno

```bash
conda env create -f environment.yml
conda activate lab03_ml_2026_01
```

El dataset debe estar en `datasets/15 atributos R0-R5.sav`. Si
se mueve, ajustar la ruta en `config/paths.yaml`.

### 17.2 Ejecucion completa

```bash
python src/main.py --keep-estimators
```

Esto ejecuta:

1. EDA (`outputs/figures/eda/`)
2. 36 experimentos (6 targets x 6 modelos) con validacion
   cruzada anidada (`outputs/tables/`, `outputs/confusion_matrices/`,
   `outputs/per_class/`)
3. Analisis avanzado: learning curves, feature importance,
   bootstrap CI, tests de significancia
   (`outputs/figures/analysis/`, `outputs/tables/`)
4. Plots comparativos (`outputs/figures/experiments/`)
5. Reportes: CSV, JSON, LaTeX, PDF
   (`outputs/tables/`)

Tarda ~45 segundos en un MacBook Pro con n_jobs=-1.

### 17.3 Ejecuciones parciales

```bash
python src/main.py --targets GDS_R3           # solo un target
python src/main.py --skip-eda                  # saltar EDA
python src/main.py --skip-analysis             # saltar bootstrap y tests
python src/main.py --skip-plots                # sin figuras (mas rapido)
```

### 17.4 Comparacion contra mejoras

```bash
python src/compare_improvements.py
python src/plot_comparisons.py
```

Genera `outputs/comparisons/comparison_baseline_vs_improvements.csv`
y figuras asociadas.

### 17.5 Reproducibilidad

- `outer_random_state=42`, `inner_random_state=123` (en
  `config/paths.yaml`).
- LogisticRegression y DecisionTreeClassifier reciben
  `random_state`.
- SVC no usa `random_state` salvo con `probability=True` (que
  no usamos).
- DummyClassifier usa `random_state` para reproducibilidad de
  las predicciones.

---

## 18. Estructura de `outputs/`

```text
outputs/
|-- advertencias.txt
|-- tables/
|   |-- resumen_resultados.csv              # tabla principal (6 x 6 = 36 filas)
|   |-- resultados_detallados.json          # todo en JSON
|   |-- resultados_experimentos.tex         # LaTeX
|   |-- resultados_experimentos.pdf         # PDF con tablas y figuras
|   |-- distribucion_clases.csv             # distribucion de cada target
|   |-- bootstrap_ci_95.csv                 # IC bootstrap por modelo y target
|   |-- significance_tests.csv              # Wilcoxon + McNemar-Yates
|   |-- low_support_classes.csv             # clases con n < 10
|   `-- zero_recall_classes.csv             # clases con recall = 0
|-- confusion_matrices/                    # 36 CSVs con la matriz por (target, modelo)
|-- per_class/                              # metricas por clase para cada (target, modelo)
|-- figures/
|   |-- eda/
|   |   |-- class_distribution.png
|   |   |-- feature_correlation.png
|   |   |-- target_relationships.png
|   |   `-- eda_summary.csv
|   |-- analysis/
|   |   |-- learning_curves.png
|   |   |-- feature_importance_{target}.png
|   |   |-- hyperparam_stability_{target}.png
|   |   |-- decision_tree_{target}.png
|   |   |-- per_fold_f1_distribution.png
|   |   |-- bootstrap_ci_forest_95.png
|   |   |-- significance_heatmap_wilcoxon.png
|   |   |-- significance_heatmap_mcnemar.png
|   |   `-- zero_recall_heatmap.png
|   `-- experiments/
|       |-- f1_macro_heatmap.png
|       |-- balanced_accuracy_heatmap.png
|       |-- metrics_heatmap_{target}.png
|       |-- confusion_matrices.png
|       |-- icn_comparison.png
|       |-- delta_sesgo.png
|       |-- svm_kernel_comparison.png
|       |-- class_distribution_report.png
|       `-- summary_dashboard.png
|-- comparisons/
|   |-- comparison_baseline_vs_improvements.csv
|   |-- f1_comparison_baseline_vs_improvements.png
|   `-- (otras figuras del analisis de mejoras)
`-- estimator_cache/                        # estimadores por (target, modelo) si --keep-estimators
```

---

## 19. Aclaraciones sobre diseno y decisiones

### 19.1 Por que la version del PDF del ICN no es la unica

El PDF (slide 40) define el ICN como una suma ponderada de
metricas crudas. Esa formula es comparable entre targets pero
esta dominada por la magnitud absoluta del target, no por las
diferencias entre modelos. Para ordenar modelos dentro de un
mismo target (caso de uso tipico), se normaliza cada componente
min-max entre los modelos de ese target. **El laboratorio reporta
ambas variantes** (`icn_raw` y `icn`) y documenta la diferencia.
En las tablas se llaman `ICN*` (crudo) y `ICN` (normalizado).

### 19.2 Por que GDS se reporta como caso exploratorio

El enunciado (slide 23) dice:

> Si una clase tiene solo 1 ejemplo, no es posible garantizar
> validacion estratificada para esa clase. El informe debe
> declarar esta limitacion y justificar si se agrupa, se excluye
> del analisis o se reporta como caso exploratorio.

En `GDS` la clase 7 tiene 2 ejemplos, lo que fuerza k_outer=2.
No es 1, pero igualmente k=2 no es una validacion cruzada
robusta. La opcion "caso exploratorio" es la que mejor
refleja el estado del conocimiento sobre este target: se
reportan las metricas como descriptivas, no como
concluyentes.

### 19.3 Por que no se aplica oversampling

El PDF slide 29 lista SMOTE, undersampling y oversampling como
"decisiones metodologicas aceptables". No se implementaron
porque:

- El `class_weight="balanced"` ya compensa parcialmente.
- SMOTE en datos binarios con n_min=2 (clase 7 de GDS) genera
  ejemplos sinteticos ruidosos.
- El laboratorio se enfoca en clasificadores fundamentales; el
  oversampling es ortogonal.

Si en el futuro se quiere explorar, `imblearn.over_sampling.SMOTE`
se integra trivialmente como un paso del Pipeline.

### 19.4 Por que se persiste el cache de estimadores

`run_nested_cv` reentrena los 30 experimentos cada vez. Si se
quiere calcular tests de significancia, el analisis anterior
reentrenaba todo de nuevo (issue conocido). Con `--keep-estimators`
los `best_estimator_` de cada fold se guardan en
`outputs/estimator_cache/`, y los tests de significancia leen de
disco. Esto desacopla el costo de los tests del costo de los
experimentos.

### 19.5 Por que se usa `reportlab` para el PDF

La version anterior generaba un PDF binario a mano. No soportaba
tildes, no tenia imagenes, y la presentacion era texto
monoespaciado. `reportlab` (ya incluido en el environment) es
trivial de agregar y produce un documento de verdad con tablas
estilizadas, imagenes y tildes correctas.

---

## 20. Notas para el companero que escribe el informe

- La **portada** debe incluir titulo, asignatura, profesor,
  integrantes (Gael Ortega, Matias Vidal), carrera y correos
  (no estan en este README; agregarlos desde otra fuente).
- Las **figuras** ya estan en `outputs/figures/`; el informe las
  referencia con su ruta relativa.
- Las **tablas** ya estan en `outputs/tables/`. El LaTeX en
  `resultados_experimentos.tex` se puede compilar directamente.
- Los **numeros** que aparecen en este README son los de la
  corrida final. Si se reejecuta `python src/main.py` con los
  mismos seeds, los numeros deberian coincidir al menos hasta
  la tercera cifra decimal.
- **Orden sugerido del informe**: Introduccion, Marco teorico
  (breve, con las formulas de LR, SVM, Arbol y K-NN),
  Metodologia (validacion anidada, grillas, metricas), Resultados
  (secciones 6-10 de este README), Discusion (limitaciones +
  respuestas a las 9 preguntas del slide 49), Conclusiones,
  Codigo y reproducibilidad (seccion 17).
