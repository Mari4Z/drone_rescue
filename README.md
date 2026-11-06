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

Para iniciar el seguimiento de datos grandes con DVC, agrega las salidas reales
cuando existan:

```bash
dvc add artifacts/q_tables artifacts/suites artifacts/results
git add artifacts/*.dvc artifacts/.gitignore
```

Luego configura un remoto DVC compartido antes de usar `dvc push` o `dvc pull`.
No se ha configurado un remoto en esta rama.

## API de simulación

La API ejecuta el mismo ambiente Python versionado; no delega transiciones ni
física al cliente. Con las dependencias instaladas, inicia Uvicorn localmente:

```bash
python -m api.main
```

`GET /catalog` combina runs de MLflow con manifiestos locales, métricas,
semillas, duración, pasos, versiones de modelo y diferencias de configuración
contra `configs/drone_rescue.yaml`. `POST /load`
acepta `{"run_id": "<run_id>"}` y verifica los SHA-256 del modelo, la
configuración y la normalización antes de activar la Q-table junto con el
ambiente resuelto original. `POST /evaluate` ejecuta la suite pública fijada
por `DRONE_RESCUE_EVALUATION_SUITE` y guarda un resultado con nombre único en
`artifacts/results/`. `POST /episode/reset` inicia un episodio; `GET
/episode/step?action=4` avanza un paso manual y devuelve posición, telemetría,
recompensa, causa y modo. Los modos `aleatorio`, `referencia` y `evaluacion`
también están disponibles; `entrenamiento` y `replay` responden HTTP 501 hasta
que exista una implementación real. `GET /health` y `GET /version` reportan
estado y versiones servidas.

Configura `DRONE_RESCUE_CORS_ORIGINS` como lista separada por comas. En
producción define `DRONE_RESCUE_ENV=production`,
`DRONE_RESCUE_SSL_CERTFILE` y `DRONE_RESCUE_SSL_KEYFILE`; Uvicorn termina TLS
directamente y la aplicación rechaza solicitudes que no usen HTTPS. Los
orígenes CORS de producción también deben usar HTTPS. `MLFLOW_UI_URL` permite
configurar un enlace navegable al UI; sin un UI HTTP(S), el catálogo devuelve
`mlflow_link: null` y reporta por separado el URI de tracking y el URI del run.
La API solo puede cargar runs cuyos artefactos MLflow y manifiestos sean
accesibles desde el servidor.

## Familia de cuadrícula (grid-drone-v1)

`cli.py` opera una familia aparte: cuadrícula discreta y misión de A a B, sin física ni
batería. No es compatible con `drone-rescue-v2` y no resuelve la misión completa; el detalle
y sus límites están en [`docs/familia_cuadricula.md`](docs/familia_cuadricula.md).

```bash
python cli.py envs
python cli.py train --algo dqn --steps 1500000 --out models/grid_dqn.pt
python cli.py eval --model models/grid_dqn.pt --episodes 100
python cli.py run --model models/grid_dqn.pt --web
```

## Arquitectura

- `api/main.py`: API FastAPI/Uvicorn para catálogo, carga, evaluación y pasos.
- `envs/`: entorno Gymnasium canónico y reglas físicas de la simulación.
- `agents/`: agente Q-learning tabular y DQN exploratorio de `drone-rescue-v2`.
- `envs/grid_drone_env.py`, `agents/grid/`, `viewer/`, `cli.py`: familia `grid-drone-v1`.
- `env/` y `agent/`: adaptadores para preservar imports históricos.
- `pipelines/train.py`: entrenamiento MLflow, evaluación greedy, checkpoint,
  Model Registry y manifiesto; `train.py` conserva la ruta CLI anterior.
- `visualize.py`, `plot3d.py`: vistas de depuración y visualización estática.
- `configs/`: configuración YAML efectiva del ambiente y recompensa.
- `pipelines/`: entrenamiento y evaluación; Prefect todavía está pendiente.
- `tests/`: pruebas del contrato actual del entorno e imports.
- `web/`: aplicación Vite, TypeScript y Three.js desplegable en Vercel.
- `manifests/`: manifiestos por run con hashes, URIs y runtime.
- `artifacts/`: destinos DVC para Q-tables, suites y resultados.
- `docs/`: auditoría y documentación técnica.

La simulación Python es la autoridad para estados y transiciones. La interfaz
Three.js solo representa el estado comunicado por la API; la
velocidad de animación no modificará el paso físico.

## Pruebas

```bash
python -m pytest
```

## Estado frente al producto final

La auditoría en [`docs/auditoria.md`](docs/auditoria.md) registra lo que existe,
lo preparado y lo que falta. No se encontró el PDF de requisitos en la rama
revisada, así que falta contrastar este inventario con el documento oficial.
