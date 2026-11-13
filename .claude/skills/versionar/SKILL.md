---
name: versionar
description: Qué hay que hacer para versionar un cambio, un experimento, un modelo o una entrega de Drone Rescue según la guía del proyecto de MLOps. Úsala al entrenar, cambiar el ambiente o la recompensa, publicar un modelo, hacer rollback, preparar una release, o cuando pregunten si algo quedó versionado.
---

# Versionar en Drone Rescue

Sigue la guía del curso (secciones 09 a 17). El detalle y la tabla de evidencia están en `docs/versionamiento.md`.

## 1. Identifica qué cambió

| Si cambió... | Debes hacer |
|---|---|
| Código sin efecto en el modelo | Commit en una rama `feature/*` y pruebas |
| Física, sensores, límites o `dt` | Subir `ENVIRONMENT_VERSION` y actualizar `docs/ficha_ambiente.md` |
| Recompensa | Subir `reward.version` en el YAML; los retornos dejan de ser comparables |
| Dimensiones o significado de observaciones o acciones | **Familia nueva**: no reutilizar modelos anteriores. Nunca cambies una familia existente |
| Escalado de entradas | Subir `FEATURES_VERSION` en `agents/dqn.py` |
| Mapas, obstáculos, viento, batería | Nuevo YAML en `configs/levels` o `configs/experiments`; un experimento cambia una sola dimensión |
| Suite de evaluación | Nueva `version` de la suite; no compares con resultados de otra versión |
| Criterios de selección | Nueva `version` en `configs/selection_criteria.yaml`, **antes** de ver resultados |
| Dependencias | Fijarlas en `requirements.txt` o `web/package-lock.json` |

Familias vigentes: `grid-drone-v1`, `grid-drone-v2`, `grid-drone-v3`, `drone-rescue-v2` y `drone-rescue-v3`. Ver `docs/versionamiento.md`. Los runs de `grid-drone-v2` y `grid-drone-v3` se entrenan con `pipelines/train_grid.py`, que deja el mismo rastro que `train_ppo.py`.

## 2. Entrenar un modelo

1. Escribe la hipótesis en un perfil de `configs/training/`, antes de entrenar.
2. Haz commit de todo. `train_ppo` rechaza un árbol con cambios, porque el commit no identificaría el código.
3. Entrena desde la raíz del repo:

   ```bash
   export MLFLOW_TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db
   .venv/bin/python -m pipelines.train_ppo --profile configs/training/<perfil>.yaml --seed <n>
   ```

4. No uses `--allow-dirty` para evidencia: es solo para pruebas y marca el run con `git_dirty=true`.
5. Para comparar finalistas, usa tres semillas de entrenamiento (11, 23 y 37) con el mismo presupuesto.

Cada run deja: run en MLflow, `manifests/<run_id>.json` y `artifacts/models/<run_id>/` con pesos, configuración, contrato y evaluación.

## 3. Conservar evidencia

Haz commit de los manifiestos nuevos y actualiza los punteros de DVC:

```bash
.venv/bin/dvc add artifacts/models artifacts/results artifacts/mlflow artifacts/q_tables
git add manifests artifacts/*.dvc artifacts/.gitignore
```

Si cambió el ambiente, la recompensa, la suite o el modelo oficial, regenera el dataset y versiónalo (`docs/dataset.md`):

```bash
.venv/bin/python -m pipelines.build_dataset
.venv/bin/dvc add data && git add data.dvc
```

Si hay remoto configurado, ejecuta `.venv/bin/dvc push`. Consulta `docs/versionamiento.md`: si no lo hay, los modelos solo se pueden recuperar desde la release de GitHub.

**Conserva los intentos fallidos y rechazados.** Nunca borres un run, manifiesto o publicación porque salió mal.

## 4. Seleccionar y publicar

Usa el pipeline, no publiques a mano:

```bash
export AIRFLOW_HOME=$PWD/.airflow AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags AIRFLOW__CORE__LOAD_EXAMPLES=False
.venv-airflow/bin/airflow dags test drone_rescue_pipeline \
  --conf '{"profile": "configs/training/<perfil>.yaml", "seed": 11, "run_id": "<run_id o vacío>"}'
```

Registrar un candidato **no** lo promueve. Solo `7_publicar` lo hace, si cumple `configs/selection_criteria.yaml`. Cada publicación es una versión nueva en `manifests/published/<nivel>/`.

Para restaurar la versión anterior y comprobarla:

```bash
.venv/bin/python -m pipelines.pipeline_steps rollback --level <nivel>
```

## 5. Release y despliegue

```bash
git tag -a <version> -m "<qué entrega>"
git push origin <rama> <version>
```

Sube el paquete de modelos a la release de GitHub, haz `dvc push` y si cambió el modelo oficial (una publicación nueva en `manifests/published/`), publícalo en Hugging Face con el mismo tag; allí solo va el modelo oficial (lo pide el profesor):

```bash
.venv/bin/python scripts/publish_huggingface.py --repo Mari4Z/drone-rescue --tag <version>
```

Comprueba desde la URL publicada: catálogo, carga de un modelo antiguo y del recomendado, y una evaluación completa. `GET /version` debe mostrar el commit y el modelo que se sirven.

## 6. Antes de decir que algo quedó versionado

Comprueba en vez de suponer:

- [ ] `git status` limpio, y el commit está en GitHub.
- [ ] El manifiesto tiene `git_dirty: false` y su `model_sha256` coincide con el archivo.
- [ ] `.venv/bin/python -m pytest -q` pasa.
- [ ] El modelo se puede recuperar **fuera de esta carpeta**: `dvc pull` o la release (y el oficial, en Hugging Face), no solo `artifacts/` local.
- [ ] La afirmación se apoya en métricas de misión de la suite, no en una demostración ni solo en el retorno.
- [ ] En niveles sin viento, los episodios de la suite son idénticos entre semillas: 100 % significa un episodio repetido.

Di con claridad lo que no se pudo comprobar.
