# Auditoría de alcance frente a los requisitos del proyecto

## Alcance de esta auditoría

El repositorio no contiene el PDF de requisitos en ninguna de sus ramas
revisadas. Por ello, este reporte contrasta el código visible y el producto
final descrito en la solicitud; no puede afirmar cumplimiento literal del PDF.

## Ya existe en la rama `1Experiment`

- Entorno Gymnasium discreto 3D, agente tabular Q-learning y entrenamiento.
- Visualización textual y generación de una imagen 3D estática.
- La dinámica del entorno se mantuvo sin cambios en esta adecuación.

## Preparado en esta adecuación

- Paquetes `envs/` y `agents/` como imports canónicos; `env/` y `agent/`
  conservan imports compatibles.
- Directorios `configs/`, `pipelines/`, `tests/`, `web/`, `manifests/` y
  `docs/` creados.
- Plantilla de manifiesto con pesos, SHA-256, normalización, versión del
  entorno, configuración resuelta, semillas, métricas, commit y run_id.
- Dependencia MLflow fijada y backend SQLite local de desarrollo configurado;
  la validación de esta adecuación queda registrada como un run.
- DVC inicializado con ubicaciones versionadas para Q-tables, suites y
  resultados. Aún no hay artefactos grandes ni remoto DVC configurado.
- Variables locales de ejemplo sin secretos y reglas de exclusión de archivos
  temporales, entornos virtuales, frontend y salidas de entrenamiento.

## Registro de validación de esta adecuación

- Hipótesis: `envs/` y `agents/` pueden ser los imports canónicos manteniendo
  compatibles los imports antiguos y el comportamiento del entorno.
- MLflow `run_id`: `96565e7e6915424c9a6744c7acacdce8`.
- Evidencia: 4 pruebas aprobadas, 0 fallidas y `dvc status` limpio.
- Este run registra la validación estructural; no representa un entrenamiento.

## Pendiente para el producto final descrito

- API FastAPI y contrato entre la API y la interfaz.
- Web Vite/TypeScript/Three.js y despliegue Vercel; la animación todavía no
  consume estado real del entorno.
- Instrumentar `train.py` para registrar runs, métricas y modelos en MLflow;
  el run actual solo registra la validación estructural, no un entrenamiento.
- Registro/alias/versionado y recuperación de modelos desde MLflow.
- Prefect para orquestar el pipeline.
- Flujo de guardado/recuperación de modelos con manifiesto validado y
  verificación SHA-256.
- Prueba automatizada de rollback de modelo.
- Remoto DVC compartido y reglas para suites/resultados concretos.
- Comparación requisito por requisito con el PDF oficial, cuando se incorpore
  al repositorio o se indique su ubicación.
