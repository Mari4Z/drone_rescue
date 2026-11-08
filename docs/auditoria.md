# Auditoría de alcance frente a los requisitos del proyecto

## Alcance de esta auditoría

El repositorio no contiene el PDF de requisitos en ninguna de sus ramas
revisadas. Por ello, este reporte contrasta el código visible y el producto
final descrito en la solicitud; no puede afirmar cumplimiento literal del PDF.

## Existía en la rama `1Experiment`

- Entorno Gymnasium de posición discreta, agente tabular Q-learning y entrenamiento.
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
- Contrato de misión y dinámica continua en la ficha del ambiente; parámetros de
  simulación y recompensa en YAML.
- DVC inicializado con ubicaciones versionadas para Q-tables, suites y
  resultados. Aún no hay artefactos grandes ni remoto DVC configurado.
- Entrenamiento `pipelines/train.py` con perfiles N1–N3, runs MLflow, curvas de
  entrenamiento y evaluación separadas, checkpoints `.npy`, snapshots, manifiestos,
  reanudación por run padre y registro de Q-tables en Model Registry.
- Evaluación reproducible desde manifiesto con verificación SHA-256 de checkpoint
  y snapshot; una prueba repite los resultados deterministas por semilla.
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
- Probar entrenamiento prolongado para medir convergencia y ajustar presupuestos
  por nivel; las pruebas de esta adecuación usan pocos episodios y no demuestran
  aprendizaje exitoso.
- Configurar servidor/registry remoto y alias de producción; el backend actual es
  MLflow local con SQLite.
- Prefect para orquestar el pipeline.
- Prueba automatizada de rollback de modelo.
- Remoto DVC compartido y reglas para suites/resultados concretos.
- Comparación requisito por requisito con el PDF oficial, cuando se incorpore
  al repositorio o se indique su ubicación.
