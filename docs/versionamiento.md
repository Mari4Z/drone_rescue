# Versionamiento

Cómo cumple el proyecto las secciones 09 a 12 de la guía. La skill `.claude/skills/versionar` resume los pasos.

## Familias de modelos

Un modelo solo se ejecuta en su familia. Si cambian las dimensiones o el significado de observaciones o acciones, se crea una familia nueva.

| Familia | Dinámica y misión | Acciones | Algoritmos | Se sirve por la API |
|---|---|---|---|---|
| `grid-drone-v1` | Cuadrícula; ir de A a B | 6 movimientos | Q-learning tabular, DQN | No; se usa `cli.py` |
| `drone-rescue-v2` | Física continua; recoger, entregar y volver | 6 fuerzas y soltar | Q-learning tabular, PPO | Sí |
| `drone-rescue-v3` | Misma física, reglas y recompensa que v2 | 10 velocidades objetivo y sostener | PPO | Sí |

**Por qué existe v3.** En v2, soltar fuera de la zona resta 0.5, y la política aprende a no intentarlo nunca (probabilidad medida: 0.0), de modo que nunca descubre la entrega. La guía sugiere usar estabilización convencional y que el agente decida velocidades objetivo. En v3 un estabilizador convierte la orden en fuerzas y decide cada 1 s (2 pasos físicos). La entrega se activa por proximidad y estabilidad: al sostener posición dentro de la zona y a baja velocidad. El código está en `envs/stabilized_drone_env.py`.

## Elementos y evidencia

| Elemento (guía, sección 10) | Dónde queda | Identidad |
|---|---|---|
| Código | Git | Commit completo en cada run y manifiesto; tag de release |
| Ambiente | `envs/`, `docs/ficha_ambiente.md` | `ENVIRONMENT_VERSION`, `STABILIZER_VERSION`, `model_family` |
| Recompensa | `configs/drone_rescue.yaml` | `reward.version` |
| Escenarios y datos | `configs/levels`, `configs/experiments`, `random_columns` | YAML en Git; no hay dataset, las experiencias se generan con estos escenarios y las semillas registradas |
| Configuración | Perfil YAML y `config.json` por run | `config_sha256` |
| Experimento | MLflow | `run_id`, hipótesis, presupuesto, semilla y estado; se conservan los intentos fallidos |
| Modelo | Registry, `artifacts/models/<run_id>/` | `model_sha256`, contrato en `normalization.json` |
| Evaluación | `evaluacion.json`, `artifacts/results/` | Suite con id, versión y SHA-256; resultados por episodio |
| Pipeline | `dags/`, `pipelines/` | Commit; estado de cada etapa en `artifacts/results/pipeline/` |
| Despliegue | `GET /version`, `manifests/published/` | Commit y modelo servidos; publicaciones sin sobrescribir |

## Paquete de un modelo (sección 11)

| # | Contenido obligatorio | Dónde |
|---|---|---|
| 1 | Pesos y SHA-256 | `model.zip`; `model_sha256` |
| 2 | Orden, dimensiones, unidades, escalado y clipping | `normalization.json`, `input_contract` en `agents/ppo.py` |
| 3 | Ambiente compatible | `environment_version`, `model_family`, `git_commit`, `runtime_ref` |
| 4 | Perfil resuelto y hash; reward, escenarios y dependencias | `config.json`, `config_sha256`, `requirements_sha256` |
| 5 | Run, modelo, presupuesto efectivo y semillas | `run_id`, `registry_uri`, `effective_timesteps`, `seed` |
| 6 | Suite y resultados por caso | `evaluacion.json`, `evaluation_suite_sha256` |
| 7 | Origen | `parent_run_id` (`null` si es desde cero) |
| 8 | Optimizador y contador | `final.zip` (Stable-Baselines3 incluye ambos) |

`model.zip` es el mejor modelo durante el entrenamiento, elegido por éxito, entrega y pasos, sin mirar el retorno. `final.zip` es el último checkpoint del run.

## Modelos históricos

`pipelines/import_historical.py` registra modelos anteriores a este pipeline con `provenance: historico`. Conservan el intento, su hash y una nueva evaluación, pero su commit es el más cercano al entrenamiento y sus parámetros de entrenamiento no quedaron registrados. No son runs reproducibles ni evidencia para seleccionar.

## Recuperar los modelos

Los pesos no van en Git. `artifacts/models` está en DVC (`artifacts/models.dvc`) y cada entrega publica además el paquete en una release de GitHub:

```bash
gh release download <tag> --pattern 'drone-rescue-modelos-*.tar.gz' --pattern SHA256SUMS
shasum -a 256 -c SHA256SUMS
tar -xzf drone-rescue-modelos-*.tar.gz      # crea artifacts/models y artifacts/q_tables
.venv/bin/python scripts/verify_packages.py  # compara cada archivo con el SHA-256 de su manifiesto
```

Con el remoto de DagsHub configurado (`scripts/setup_dagshub.sh`), `dvc pull` también recupera modelos, tablas Q, resultados y el almacén de MLflow.

Los pesos originales de `exp-001` a `exp-004` (5 de noviembre) se perdieron. Se reentrenaron con el mismo commit, perfil, semilla y suite (`scripts/reproduce_exp.py`) y quedaron en `artifacts/q_tables/reproducciones/`. Las métricas reproducidas coinciden con `docs/decision.md`; solo `exp-003` coincide además bit a bit, así que se restauró con su nombre original. La comparación completa está en `artifacts/results/reproduccion_exp_001_004.json`, y `verify_packages.py` acepta esas reproducciones en lugar de los originales.

## Reproducir

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
export MLFLOW_TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db
.venv/bin/python -m pipelines.train_ppo --profile configs/training/ppo_v3_n1.yaml --seed 11
```

Airflow usa su propio entorno, para no alterar las dependencias de los modelos:

```bash
python3.12 -m venv .venv-airflow
.venv-airflow/bin/pip install "apache-airflow==3.4.0" \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.12.txt
export AIRFLOW_HOME=$PWD/.airflow AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags AIRFLOW__CORE__LOAD_EXAMPLES=False
.venv-airflow/bin/airflow db migrate
.venv-airflow/bin/airflow dags test drone_rescue_pipeline --conf '{"profile": "configs/training/ppo_v3_n1.yaml", "seed": 11}'
```
