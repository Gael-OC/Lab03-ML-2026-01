# Lab03-ML-2026-01

Implementacion base del Laboratorio 03: clasificadores fundamentales sobre el dataset `.sav` de 15 atributos binarios.

## Alcance

- Se replican los seis experimentos del PDF: `GDS`, `GDS_R1`, `GDS_R2`, `GDS_R3`, `GDS_R4` y `GDS_R5`.
- Se implementa validacion cruzada anidada.
- Se implementa solo Regresion Logistica.
- SVM lineal, SVM RBF, Arbol de decision y K-NN quedan creados como pendientes comentados en `src/models.py`.
- Las tablas reportan `No implementado` para los modelos pendientes.

## Configurar dataset

Editar `config/paths.yaml` si el archivo `.sav` se mueve de ubicacion:

```yaml
dataset:
  path: "datasets/15 atributos R0-R5.sav"
```

## Crear ambiente conda

```bash
conda env create -f environment.yml
conda activate lab03_ml_2026_01
```

## Ejecutar experimentos

```bash
python src/main.py
```

Tambien se puede ejecutar un subconjunto de objetivos:

```bash
python src/main.py --targets GDS_R2 GDS_R5
```

## Salidas generadas

- `outputs/tables/resultados_experimentos.tex`: tablas en LaTeX por experimento.
- `outputs/tables/resultados_experimentos.pdf`: tablas en PDF.
- `outputs/tables/resumen_resultados.csv`: resumen tabular de todos los modelos.
- `outputs/tables/resultados_detallados.json`: resultados completos.
- `outputs/tables/distribucion_clases.csv`: soporte por clase.
- `outputs/confusion_matrices/`: matrices de confusion agregadas de Regresion Logistica.
- `outputs/per_class/`: precision, recall y F1 por clase.
- `outputs/advertencias.txt`: advertencias metodologicas de validacion.

## Nota metodologica

Para cada objetivo se usa `k_outer = min(5, n_min)`, donde `n_min` es el soporte de la clase menos frecuente. En el ciclo interno se usa la grilla reducida de Regresion Logistica indicada en el PDF: `C = [0.01, 0.1, 1, 10]`.

Cuando una clase queda con solo un ejemplo dentro del entrenamiento externo, no existe una particion interna estratificada valida para esa clase. En ese caso el codigo mantiene la estructura anidada y usa un `KFold` interno no estratificado, dejando la advertencia en `outputs/advertencias.txt`.
