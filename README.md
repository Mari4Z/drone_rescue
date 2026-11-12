# Drone Rescue

Proyecto académico de simulación y aprendizaje por refuerzo para una misión de
rescate con dron. El ambiente Gymnasium simula una misión física continua sobre
un mundo configurado como cuadrícula y Q-learning usa una discretización
explícita de su observación. La especificación está en
[`docs/ficha_ambiente.md`](docs/ficha_ambiente.md).

## Requisitos

- Python 3.12 o posterior. La instalación de dependencias se validó en Python
  3.14.4; Python 3.12 es la versión objetivo para desarrollo reproducible.
- Git.
- Node.js LTS y npm para desarrollar o compilar `web/`; no se necesitan para
  entrenar el agente.

## Instalación

### Windows (PowerShell)

```powershell
git clone https://github.com/Mari4Z/drone_rescue.git
cd drone_rescue
git switch feature/estructura-mlops
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux o macOS

```bash
git clone https://github.com/Mari4Z/drone_rescue.git
cd drone_rescue
git switch feature/estructura-mlops
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

El archivo `.env` es local y no debe contenerse en Git. `.env.example` solo
documenta nombres de variables y valores de desarrollo no sensibles.

## Ejecución actual

Con el entorno virtual activo:

```bash
python -m pipelines.train --profile configs/training/n1.yaml --seed 0
```

Los perfiles `configs/training/n1.yaml`, `n2.yaml` y `n3.yaml` fijan el nivel,
hipótesis, presupuesto y semillas de evaluación greedy. Cada corrida crea un run
MLflow, curvas separadas de entrenamiento y evaluación, checkpoint `.npy`,
snapshot de configuración, evaluación pública y `manifests/<run_id>.json`. El
checkpoint se registra además en el Model Registry.

Para continuar desde un run exitoso:

```bash
python -m pipelines.train --profile configs/training/n1.yaml --seed 1 \
  --parent-run-id <run_id>
```

El manifiesto permite volver a ejecutar la evaluación y verificar los SHA-256:

```bash
python - <<'PY'
from pipelines.evaluate import evaluate_from_manifest

evaluate_from_manifest(
    "manifests/<run_id>.json",
    "configs/evaluation_suite.yaml",
    "artifacts/results/reproduccion-<run_id>.json",
)
PY
```

Para producir la imagen 3D estática:

```bash
python plot3d.py
```

Por defecto crea una ruta única dentro de `outputs/`. Si indicas `--out`, la
ruta debe ser nueva; el programa se niega a sobrescribir una imagen existente.
Para abrir la ventana interactiva de Matplotlib:

```bash
python plot3d.py --show
```

El evaluador usa la suite pública fija `drone-rescue-public-validation`; el set
final reservado debe mantenerse fuera del repositorio.

Los pesos y resultados actuales tienen punteros DVC en Git. Para conectarlos a
un almacenamiento de equipo sin administrar un bucket S3, sigue
[`docs/despliegue.md`](docs/despliegue.md) para crear el remoto de DagsHub y
ejecutar `dvc push`. Los datos binarios no se suben con `git push`.

La base MLflow local y `mlruns/` tampoco se versionan con Git. Antes de
desplegar, migra sus runs y artefactos con
`scripts/migrate_mlflow_to_remote.py`, siguiendo la misma guía.

## API de simulación

La API ejecuta el mismo ambiente Python versionado; no delega transiciones ni
física al cliente. Con las dependencias instaladas, inicia Uvicorn localmente:

```bash
python -m api.main
```

`GET /catalog` combina runs de MLflow con manifiestos, métricas,
semillas, duración, pasos, versiones de modelo y diferencias de configuración
contra `configs/drone_rescue.yaml`. `POST /load`
acepta `{"run_id": "<run_id>"}` y verifica los SHA-256 del modelo, la
configuración y la normalización antes de activar la Q-table junto con el
ambiente resuelto original. `POST /evaluate` ejecuta la suite pública fijada
por `DRONE_RESCUE_EVALUATION_SUITE` y guarda el resultado como artefacto de un
nuevo run de evaluación en MLflow. `POST /episode/reset` inicia un episodio; `GET
/episode/step?action=4` avanza un paso manual y devuelve posición, telemetría,
recompensa, causa y modo. Los modos `aleatorio`, `referencia` y `evaluacion`
también están disponibles. `POST /session` inicia control manual sin modelo;
`POST /training` inicia un run asíncrono y `GET /training/{job_id}` informa
progreso; `replay` vuelve a ejecutar de forma determinista el modelo cargado
con la semilla elegida. Cada navegador recibe una sesión aislada mediante
`X-Drone-Session`. `GET /health` y `/version` reportan estado y versiones.

En producción `DRONE_RESCUE_API_TOKEN` es obligatorio; se envía en el
encabezado `X-API-Key`, no en `VITE_*`. Configura
`DRONE_RESCUE_CORS_ORIGINS` como lista separada por comas y exige orígenes
HTTPS. Tras un proxy HTTPS confiable, define
`DRONE_RESCUE_TRUST_PROXY_HTTPS=true`; si no, configura certificado y clave TLS
directamente en Uvicorn. Define `MLFLOW_TRACKING_URI` y credenciales del
servidor MLflow compartido para que catálogo, entrenamiento y carga de modelos
usen el mismo tracking remoto.

La API implementa hasta 32 sesiones de simulación concurrentes por proceso,
con expiración por inactividad. Los trabajos de entrenamiento no sobreviven a
un reinicio de la API.

## Despliegue compartido

La guía [`docs/despliegue.md`](docs/despliegue.md) describe Vercel, Render,
autenticación compartida, migración de MLflow local y remoto DVC. El Blueprint
de Render incluido usa el plan gratuito; por sus límites de suspensión y disco
efímero no garantiza disponibilidad continua. Para uptime permanente se
requiere un servicio activo de pago.

## Familias de cuadrícula (grid-drone-v1 y grid-drone-v2)

`cli.py` opera dos familias aparte, de cuadrícula discreta y misión de A a B. `grid-drone-v1`
no tiene física ni batería; `grid-drone-v2` (`DroneRescueGridStorm-v3`) añade una ciudad más
densa con B aleatoria, viento con ráfagas que empujan al dron y una batería ajustada a cada
mapa. No son compatibles con `drone-rescue-v2` ni resuelven la misión completa; el detalle,
los resultados y sus límites están en [`docs/familia_cuadricula.md`](docs/familia_cuadricula.md).

```bash
python cli.py envs
python cli.py rules --env DroneRescueGridStorm-v3
python -m pipelines.train_grid --profile configs/training/grid_storm_dqn.yaml --seed 11   # run versionado
python cli.py eval --env DroneRescueGridStorm-v3 --model artifacts/models/<run_id>/model.pt --episodes 100
python cli.py run  --env DroneRescueGridStorm-v3 --model artifacts/models/<run_id>/model.pt --seed 600000 --web
```

## Arquitectura

- `api/main.py`: API FastAPI/Uvicorn para catálogo, carga, evaluación y pasos.
- `envs/`: entorno Gymnasium canónico y reglas físicas de la simulación.
- `agents/`: agente Q-learning tabular y DQN exploratorio de `drone-rescue-v2`.
- `envs/grid_drone_env.py`, `agents/grid/`, `viewer/`, `cli.py`, `pipelines/train_grid.py`: familias `grid-drone-v1` y `grid-drone-v2`.
- `env/` y `agent/`: adaptadores para preservar imports históricos.
- `pipelines/train.py`: entrenamiento MLflow, evaluación greedy, checkpoint,
  Model Registry y manifiesto; `train.py` conserva la ruta CLI anterior.
- `visualize.py`, `plot3d.py`: vistas de depuración y visualización estática.
- `configs/`: configuración YAML efectiva del ambiente y recompensa.
- `pipelines/`, `dags/`: entrenamiento, evaluación y etapas del pipeline, orquestadas con Airflow.
- `tests/`: pruebas del contrato actual del entorno e imports.
- `web/`: aplicación Vite, TypeScript y Three.js desplegable en Vercel.
- `manifests/`: manifiestos por run con hashes, URIs y runtime.
- `artifacts/`: destinos DVC para Q-tables, suites y resultados.
- `docs/`: auditoría y documentación técnica.

La simulación Python es la autoridad para estados y transiciones. La interfaz
Three.js solo representa el estado comunicado por la API; la
velocidad de animación no modificará el paso físico.

## PPO y versionamiento

El agente de referencia es PPO con la familia `drone-rescue-v3`: el agente ordena velocidades objetivo y un estabilizador aplica las fuerzas. Qué se versiona, dónde y cómo reproducirlo está en [`docs/versionamiento.md`](docs/versionamiento.md); la skill `.claude/skills/versionar` resume los pasos.

```bash
export MLFLOW_TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db
.venv/bin/python -m pipelines.train_ppo --profile configs/training/ppo_v3_n1.yaml --seed 11
```

## Pruebas

```bash
python -m pytest
```

## Estado frente al producto final

La auditoría en [`docs/auditoria.md`](docs/auditoria.md) registra lo que existe,
lo preparado y lo que falta. No se encontró el PDF de requisitos en la rama
revisada, así que falta contrastar este inventario con el documento oficial.
