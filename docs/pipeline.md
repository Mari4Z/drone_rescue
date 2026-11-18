# Pipeline, selección, rollback y evidencia

Cumple la sección 14 de la guía. El pipeline lleva un perfil versionado hasta un modelo publicado y comprobado, deja evidencia en cada etapa, decide con criterios fijados antes de ver resultados y puede volver a la versión anterior.

| Pieza | Dónde |
|---|---|
| DAGs de Airflow | [`dags/drone_rescue_pipeline.py`](../dags/drone_rescue_pipeline.py): `drone_rescue_pipeline` (8 tareas) y `drone_rescue_rollback` (2 tareas) |
| Código de cada etapa | [`pipelines/pipeline_steps.py`](../pipelines/pipeline_steps.py) |
| Criterios de selección | [`configs/selection_criteria.yaml`](../configs/selection_criteria.yaml), versión 1.0.0 |
| Publicaciones | `manifests/published/<nivel>/vNNNN.json` (en Git) |
| Evidencia de cada ejecución | `artifacts/results/pipeline/<fecha>.json` (en DVC y en la release) |

Airflow vive en su propio entorno (`.venv-airflow`, Python 3.12) y solo orquesta: cada tarea ejecuta `python -m pipelines.pipeline_steps <etapa>` con el entorno `.venv` del proyecto, así el orquestador no altera las dependencias fijadas de los modelos.

## Arrancar el pipeline

```bash
python3.12 -m venv .venv-airflow
.venv-airflow/bin/pip install "apache-airflow==3.4.0" \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.12.txt
export AIRFLOW_HOME=$PWD/.airflow AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags AIRFLOW__CORE__LOAD_EXAMPLES=False
.venv-airflow/bin/airflow db migrate

# entrenar un candidato y recorrer las 8 etapas
.venv-airflow/bin/airflow dags test drone_rescue_pipeline \
  --conf '{"profile": "configs/training/ppo_v3_n1.yaml", "seed": 11}'
```

Parámetros de `drone_rescue_pipeline`: `profile` (perfil de entrenamiento), `seed`, `timesteps` (0 usa el presupuesto del perfil), `run_id` (usa un run existente en lugar de entrenar) y `parent_run_id` (continúa el entrenamiento desde el checkpoint de ese run). Para recorrer las etapas sobre un modelo ya entrenado: `--conf '{"profile": "configs/training/ppo_v3c_n1.yaml", "run_id": "<run_id>"}'`.

## Las 8 etapas

Cada etapa deja su estado y evidencia en el JSON de la ejecución, y falla con diagnóstico en lugar de seguir.

| # | Tarea | Qué comprueba o hace | Evidencia que deja |
|---:|---|---|---|
| 1 | `1_validar` | Carga el perfil, crea el ambiente y hace `reset`. El nivel debe estar en la suite y tener criterio de selección. | SHA-256 del perfil y de la suite, versión de los criterios y del ambiente |
| 2 | `2_probar` | Ejecuta las pruebas de reglas, observaciones, acciones y terminaciones (`tests/test_environment_contract.py` y `tests/test_dqn.py`). | Resumen de pytest y commit |
| 3 | `3_entrenar` | Entrena un candidato, desde cero o continuando un checkpoint, o toma un run existente. Un run sin checkpoint se rechaza con su diagnóstico. | `run_id`, versión en el Registry, origen |
| 4 | `4_evaluar` | Repite la suite con el modelo congelado (comprueba su SHA-256 contra el manifiesto) y exige que reproduzca la evaluación registrada. | Archivo de evaluación y éxito por caso |
| 5 | `5_registrar` | Descarga el modelo desde MLflow, compara su hash con el manifiesto y revisa el estado de la versión en el Model Registry. | URI del Registry, hash, commit y `git_dirty` |
| 6 | `6_seleccionar` | Aplica los criterios fijados y etiqueta el run con `selection_decision`. Rechazar deja el candidato y todos los motivos. | Decisión, motivos, SHA-256 de los criterios |
| 7 | `7_publicar` | Solo si fue aceptado: escribe una versión nueva en `manifests/published/<nivel>/` y mueve el alias `produccion` del Registry. Nunca sobrescribe; repetir la vigente no crea otra. | Versión publicada |
| 8 | `8_verificar` | Carga la versión publicada por la API, comprueba que el SHA-256 servido sea el publicado y corre un episodio de humo (semilla 101). | SHA-256 servido, causa y pasos del episodio |

Registrar un candidato (etapa 5) no lo promueve: solo la etapa 7 lo hace, y solo si la 6 lo acepta.

## Criterios de selección

Fijados antes de ver resultados (`configs/selection_criteria.yaml`, versión 1.0.0). Un candidato se publica solo si cumple todos; se miden en el caso de la suite que corresponde al nivel del modelo y el retorno no interviene.

| Criterio | Valor |
|---|---|
| Éxito mínimo | N1 80 %, N2 80 %, N3 70 % (en N3 el viento cambia el episodio con la semilla, así que no se espera 100 %) |
| Pasos medios máximos | 700 |
| Sin regresión | Se rechaza si tiene menos éxito que el modelo vigente; con igual éxito, gana el de menos pasos |

## Corridas reales

Cinco ejecuciones del 11 de noviembre de 2026 (hora UTC). Su evidencia está en `artifacts/results/pipeline/`.

| # | Hora | DAG | Candidato | Resultado |
|---:|---|---|---|---|
| 1 | 03:31 | `drone_rescue_pipeline` | `ppo-v3b-n1`, semilla 11 (`9a188153`), N1 | Aceptado, éxito 100 %: **publicación v0001** |
| 2 | 03:33 | `drone_rescue_pipeline` | `ppo-v3b-n2`, semilla 23 (`f0c217b8`), N2 | **Rechazado**: «éxito 0% por debajo del mínimo 80%». No se publicó nada y el candidato se conserva |
| 3 | 21:58 | `drone_rescue_pipeline` | `ppo-v3c-n1` (`cef121d1`): continuación de 1 millón de pasos de `9a188153` | Aceptado, éxito 100 %: **publicación v0002** |
| 4 | 22:09 | `drone_rescue_rollback` | N1 | **Rollback**: restaura v0001 como **v0003**; la API sirve de nuevo el SHA-256 `f544dbf0…` |
| 5 | 22:09 | `drone_rescue_pipeline` | `ppo-v3c-n1` (`cef121d1`) | Aceptado: **publicación v0004**, el modelo oficial de N1 |

Las cuatro publicaciones de N1 (`manifests/published/N1/v0001.json` a `v0004.json`) conservan el historial completo: v0001 y v0002 son promociones, v0003 es el rollback (`restored_from_version: 1`, `replaced_version: 2`) y v0004 vuelve a promover `ppo-v3c-n1`.

## Rollback

Restaura la versión inmediatamente anterior a la vigente como una versión nueva, conserva todo el historial, mueve el alias `produccion` del Registry y comprueba qué modelo sirve realmente la API.

```bash
# con Airflow
.venv-airflow/bin/airflow dags test drone_rescue_rollback --conf '{"level": "N1"}'

# sin Airflow (restaura y verifica)
.venv/bin/python -m pipelines.pipeline_steps rollback --level N1
```

Cada ejecución agrega una versión a `manifests/published/<nivel>/`: no es una comprobación sin efectos. Un nivel con una sola publicación no tiene versión anterior y el rollback falla con ese diagnóstico. Para ver qué modelo está vigente sin modificar nada: `ls manifests/published/N1/` (la versión más alta) y `GET /version` de la API.

### Sin la base de MLflow original

Con solo la release (sin `artifacts/mlflow`) el pipeline sigue funcionando, y lo que MLflow no puede hacer queda anotado en la evidencia en lugar de hacer fallar la etapa:

| Etapa | Qué pasa si este MLflow no tiene el run o el modelo registrado |
|---|---|
| 5 `register` | Comprueba el hash del paquete local (`model_source: paquete local`) y anota `registry_status: sin registro en este MLflow` |
| 6 `select` | Decide igual con los criterios; anota `selection_tag: omitida` porque no puede etiquetar el run |
| 7 `publish` | Publica igual la versión nueva; anota `registry_alias: omitido` |
| 8 `verify` | Funciona: la API carga el modelo desde el paquete local |
| Rollback | Restaura la publicación y avisa por stderr que no movió el alias `produccion` |

Se probó en una copia limpia con la release: `verify` y `rollback --level N1` terminaron bien. Las pruebas están en `tests/test_offline.py`.

## Comprobar la evidencia sin Airflow

```bash
.venv/bin/python scripts/reproduce_evaluation.py     # repite la evaluación del modelo oficial y la compara episodio por episodio
.venv/bin/python scripts/verify_packages.py          # compara cada archivo de los paquetes con el SHA-256 de su manifiesto
.venv/bin/python -m pipelines.pipeline_steps verify --level N1   # carga la publicación vigente por la API y comprueba el SHA-256 servido
```

Con la release descomprimida o `dvc pull`, los dos primeros funcionan en una copia limpia del repositorio (ver [`docs/versionamiento.md`](versionamiento.md)).

## Monitoreo mínimo

| Qué | Dónde se ve |
|---|---|
| Éxito por escenario y causas de fallo | `evaluacion.json` de cada paquete (por caso y por episodio) y `POST /evaluate` de la API |
| Latencia de inferencia | Campo `inference_latency_ms` de cada episodio |
| Errores de servicio | La web los muestra como error; un replay se identifica como replay |
| Versión servida | `GET /version`: commit del código y modelo cargado (`served_model` con su SHA-256) |

## Límites

- La etapa 2 ejecuta solo dos archivos de pruebas rápidas; la suite completa (`python -m pytest`) se corre aparte.
- La verificación de la etapa 8 usa el cliente de pruebas de FastAPI, no la URL pública.
- Los dos primeros registros de evidencia (corridas 1 y 2) son anteriores a la etapa de pruebas, que se agregó después; por eso no la incluyen.
