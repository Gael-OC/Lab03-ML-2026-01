# Plan de Implementacion del Laboratorio 03

Este repositorio replica la estructura experimental del PDF del Laboratorio 03 para los seis objetivos:
`GDS`, `GDS_R1`, `GDS_R2`, `GDS_R3`, `GDS_R4` y `GDS_R5`.

## Alcance implementado

1. Cargar el archivo `.sav` configurado en `config/paths.yaml`.
2. Usar los 15 atributos binarios indicados en el PDF y excluir `ID`.
3. Ejecutar seis experimentos independientes, uno por cada columna objetivo.
4. Implementar validacion cruzada anidada:
   - ciclo externo estratificado con `k_outer = min(5, n_min)`;
   - ciclo interno para seleccion de hiperparametros con grilla reducida;
   - seleccion por `f1_macro`;
   - reporte de promedio y desviacion estandar sobre folds externos.
5. Implementar solamente Regresion Logistica.
6. Crear los modelos SVM lineal, SVM RBF, Arbol de decision y K-NN como pendientes comentados para implementacion de los alumnos.
7. Generar tablas por experimento en LaTeX y PDF. Los modelos no implementados aparecen como `No implementado`.

## Pendiente para los alumnos

Los alumnos deben completar los pipelines y grillas de:

- SVM lineal.
- SVM con kernel RBF.
- Arbol de decision.
- K-NN.

Las plantillas estan comentadas en `src/models.py`.
