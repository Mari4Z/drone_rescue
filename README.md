# Drone Rescue

Un dron sale de BASE, recoge un suministro, lo entrega en la zona de entrega y vuelve a BASE. El ambiente (Gymnasium, física continua en un mundo de 100 × 100 × 60 m con columnas, viento y batería) vive en Python; un agente PPO (Stable-Baselines3) aprende a pilotarlo; una web en Three.js muestra el estado que le devuelve una API FastAPI. Cada experimento queda registrado en MLflow, versionado con Git y DVC, y publicado por un pipeline de Airflow que puede volver a la versión anterior. Proyecto final de MLOps, Universidad Externado de Colombia.

**Estado:** hay un modelo oficial para N1, `ppo-v3c-n1` (10 de 10 en 199 pasos). Ningún modelo resuelve N2 ni N3. Los resultados, también los de los intentos que no mejoraron, están en el [informe](docs/informe.md).

## Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Informe de decisiones, resultados, límites y contribuciones | [docs/informe.md](docs/informe.md) |
| Cumplimiento de la guía, punto por punto | [docs/auditoria.md](docs/auditoria.md) |
| Ficha del ambiente | [docs/ficha_ambiente.md](docs/ficha_ambiente.md) |
| Tres niveles de dificultad | [docs/niveles.md](docs/niveles.md) y `configs/levels/` |
| Configuraciones con hipótesis | `configs/experiments/` (exp-001 a exp-004), `configs/training/` (perfiles); cada YAML declara su hipótesis |
| Evaluaciones por caso | `evaluacion.json` de cada paquete en `artifacts/models/<run_id>/` (por caso y por episodio) y las suites `configs/*evaluation_suite.yaml` |
| Modelos, manifiestos, normalización, hashes y dependencias | [docs/versionamiento.md](docs/versionamiento.md), `manifests/`, release de GitHub |
| Pipeline, criterios de selección, rollback y corridas reales | [docs/pipeline.md](docs/pipeline.md) |
| Incidente de degradación e intervención | [docs/informe.md](docs/informe.md), sección 5 |
| Despliegue y acceso para la revisión | [docs/despliegue.md](docs/despliegue.md) |
| Experimentos de cuadrícula | [docs/familia_cuadricula.md](docs/familia_cuadricula.md) |

## Acceso para la revisión

- **Web:** https://drone-rescue-snowy.vercel.app
- **API:** la web consume una API que corre en un equipo del grupo, publicada con ngrok. Solo está disponible mientras ese equipo esté encendido; si la web muestra «API no disponible», no es un fallo de la página. `GET /health` y `GET /version` (commit del código y modelo servido) de la API sirven para comprobarla.
- **Contraseñas:** la demostración no exige clave. Si se activa, se entrega por el canal del curso y nunca por correo ([docs/despliegue.md](docs/despliegue.md)).
- **Modelos sin cuenta:** release de GitHub (`modelos-20261117`, todos los modelos) y [Hugging Face](https://huggingface.co/Mari4Z/drone-rescue) (solo el modelo oficial). Con ellos se pueden verificar y reproducir las evaluaciones sin la API (ver «Reproducir una evaluación»).
- **Experimentos y datos:** MLflow y DVC están en [DagsHub](https://dagshub.com/Mari4Z/drone_rescue), que pide una cuenta gratuita incluso para leer. La release **no incluye** la base de MLflow: levantar la API y la web en local necesita los runs (`dvc pull artifacts/mlflow`, con cuenta de DagsHub); sin ellos el catálogo queda vacío y no se puede cargar un modelo.

## Requisitos e instalación

- **Python 3.14**: con él se entrenaron todos los modelos registrados (3.14.4 y 3.14.6, según el `runtime_ref` de sus manifiestos), en macOS. Las versiones de `requirements.txt` también existen para Python 3.12 (comprobado con `pip install --dry-run`), pero no se reentrenó con él.
- **Git**, y **Node.js LTS con npm** solo para `web/` (desarrollo con Node 26.8.1 y npm 11.19.0; `web/package-lock.json` fija Vite y three.js).
- `requirements.txt` fija todas las versiones, incluidas `torch` y `stable_baselines3`. Las instrucciones de Windows y Linux no se volvieron a probar con este conjunto de dependencias; en Linux, `pip` instala por omisión PyTorch con CUDA, y para CPU conviene instalar antes la misma versión desde `https://download.pytorch.org/whl/cpu`.

### Linux o macOS

```bash
git clone https://github.com/Mari4Z/drone_rescue.git
cd drone_rescue
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Windows (PowerShell)

```powershell
git clone https://github.com/Mari4Z/drone_rescue.git
cd drone_rescue
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Para reproducir exactamente una entrega, usa su tag: `git checkout modelos-20261117`. El archivo `.env` es local y no va a Git; `.env.example` solo documenta nombres de variables y valores de desarrollo no sensibles.

## Ejecución

Con el entorno virtual activo:

```bash
python -m pytest                                  # pruebas
python -m api.main                                # API en http://127.0.0.1:8000
cd web && npm ci && npm run dev                   # web en desarrollo; usa la API local
```

**Entrenar y registrar un PPO** (crea un run en MLflow, el paquete en `artifacts/models/<run_id>/` y `manifests/<run_id>.json`; exige un árbol sin cambios pendientes):

```bash
export MLFLOW_TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db
python -m pipelines.train_ppo --profile configs/training/ppo_v3_n1.yaml --seed 11
```

**Reproducir una evaluación** a partir de un manifiesto y un paquete (de la release o de `dvc pull`); repite la suite con el modelo congelado y la compara episodio por episodio con la registrada:

```bash
python scripts/reproduce_evaluation.py            # el modelo oficial de N1
python scripts/reproduce_evaluation.py <run_id>   # otro run PPO
python scripts/verify_packages.py                 # hash de cada archivo de los paquetes contra su manifiesto
```

**Pipeline con Airflow y rollback:** [docs/pipeline.md](docs/pipeline.md).

**Recuperar los modelos** (release o DVC): [docs/versionamiento.md](docs/versionamiento.md).

### Cuadrícula (familias aparte)

`cli.py` opera tres familias de cuadrícula discreta y misión de A a B. `grid-drone-v1` no tiene física ni batería; `grid-drone-v2` (`DroneRescueGridStorm-v3`) añade una ciudad más densa con B aleatoria, viento con ráfagas y una batería ajustada a cada mapa; `grid-drone-v3` lo lleva a 200 × 200 m con tendidos eléctricos y aves, en dos versiones del ambiente: `DroneRescueGridCity-v4` (3.0.0) y `DroneRescueGridCityHard-v0` (3.1.0, con postes de 30 m, cables a 20-30 m y aves con zona de peligro). No son compatibles con `drone-rescue-v2` ni resuelven la misión completa; el detalle, los resultados y sus límites están en [docs/familia_cuadricula.md](docs/familia_cuadricula.md).

```bash
python cli.py envs
python cli.py rules --env DroneRescueGridCityHard-v0
python -m pipelines.train_grid --profile configs/training/grid_storm_dqn.yaml --seed 11   # run versionado
python cli.py eval --env DroneRescueGridStorm-v3 --model artifacts/models/<run_id>/model.pt --episodes 100
python cli.py run  --env DroneRescueGridStorm-v3 --model artifacts/models/<run_id>/model.pt --seed 600000 --web
```

### Histórico: Q-learning tabular

La primera versión del agente usaba Q-learning con una discretización explícita de la observación (`pipelines/train.py`, perfiles `configs/training/n1.yaml` a `n3.yaml`). Nunca completó la misión (0 % en exp-001 a exp-004, ver [docs/decision.md](docs/decision.md)) y se conserva como línea base:

```bash
python -m pipelines.train --profile configs/training/n1.yaml --seed 0
python plot3d.py          # imagen 3D estática de la ruta; crea una ruta nueva en outputs/
```

## API de simulación

La API ejecuta el mismo ambiente Python versionado; no delega transiciones ni física al cliente. `GET /catalog` combina runs de MLflow con manifiestos, métricas, semillas, duración, pasos, versiones de modelo y diferencias de configuración contra `configs/drone_rescue.yaml`. `POST /load` acepta `{"run_id": "<run_id>"}` y verifica los SHA-256 del modelo, la configuración y la normalización antes de activar el modelo junto con el ambiente resuelto original. `POST /evaluate` ejecuta la suite pública fijada por `DRONE_RESCUE_EVALUATION_SUITE` y guarda el resultado como artefacto de un nuevo run de evaluación en MLflow. `POST /episode/reset` inicia un episodio; `GET /episode/step?action=4` avanza un paso manual y devuelve posición, telemetría, recompensa, causa y modo. Los modos `aleatorio`, `referencia` y `evaluacion` también están disponibles. `POST /session` inicia control manual sin modelo; `POST /training` inicia un run asíncrono y `GET /training/{job_id}` informa progreso; `replay` vuelve a ejecutar de forma determinista el modelo cargado con la semilla elegida. Cada navegador recibe una sesión aislada mediante `X-Drone-Session`. `GET /health` y `GET /version` reportan estado y versiones (`/version` fija el commit al arrancar). Hasta 32 sesiones concurrentes por proceso, con expiración por inactividad; los trabajos de entrenamiento no sobreviven a un reinicio de la API.

## Variables de entorno

| Variable | Para qué | Por omisión |
|---|---|---|
| `DRONE_RESCUE_ENV` | `development` o `production`; en producción la API exige clave, HTTPS y orígenes CORS HTTPS | `development` |
| `DRONE_RESCUE_API_HOST`, `DRONE_RESCUE_API_PORT` | Dirección y puerto de la API | `127.0.0.1`, `8000` |
| `DRONE_RESCUE_API_TOKEN` | Clave compartida que la API pide en `X-API-Key` (todo salvo `/health` y `/version`); obligatoria en producción. Nunca en `VITE_*` | vacía: sin clave |
| `DRONE_RESCUE_CORS_ORIGINS` | Orígenes permitidos, separados por comas | `http://localhost:5173` y similares |
| `DRONE_RESCUE_TRUST_PROXY_HTTPS` | Confiar en `X-Forwarded-Proto` tras un proxy HTTPS de confianza | no |
| `DRONE_RESCUE_TRAINING_MAX_TIMESTEPS` | Tope de pasos de un entrenamiento lanzado desde la API | 300 000 (`serve_public.sh` usa 100 000) |
| `DRONE_RESCUE_EVALUATION_SUITE` | Suite que usa `POST /evaluate` | `configs/finalist_evaluation_suite.yaml` |
| `DRONE_RESCUE_GIT_COMMIT` | Fuerza el commit que reporta `/version` | `git rev-parse HEAD` al arrancar |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | Servidor MLflow; con DagsHub, el token va como contraseña | `sqlite:///artifacts/mlflow/mlflow.db` |
| `MLFLOW_UI_URL` | Enlace a la interfaz de MLflow que muestra la web | vacía |
| `VITE_API_BASE_URL` | URL HTTPS de la API para la web (es pública: sin secretos) | `http://127.0.0.1:8000` en desarrollo |

## Arquitectura

```
Navegador ──► Vercel: web estática (Vite, TypeScript, Three.js)
                  │  HTTPS (VITE_API_BASE_URL)
                  ▼
            API FastAPI ──► ambiente Gymnasium (Python) + modelo cargado
            (equipo del grupo + ngrok)
                  │
                  ├─► MLflow: runs, métricas y modelos (SQLite local; copia en DagsHub)
                  └─► manifests/ y artifacts/: Git + DVC (DagsHub), releases de GitHub, Hugging Face (solo el oficial)
```

- `api/main.py`: API FastAPI/Uvicorn para catálogo, carga, evaluación y pasos.
- `envs/`: ambientes Gymnasium y reglas físicas; `envs/grid_drone_env.py` tiene las familias de cuadrícula.
- `agents/`: PPO, Q-learning tabular y DQN (`agents/grid/` para la cuadrícula).
- `pipelines/`: entrenamiento (`train_ppo.py`, `train_grid.py`, `train.py`), evaluación y etapas del pipeline (`pipeline_steps.py`); `dags/`: DAGs de Airflow.
- `configs/`: perfiles, niveles, experimentos, suites de evaluación y criterios de selección.
- `manifests/`: un manifiesto por run, con hashes, URIs y runtime; `manifests/published/` guarda las publicaciones.
- `artifacts/`: modelos, resultados, tablas Q y MLflow (en DVC, no en Git).
- `web/`: aplicación Vite, TypeScript y Three.js ([web/README.md](web/README.md)).
- `viewer/`, `cli.py`: visores 3D y línea de comandos de las familias de cuadrícula.
- `scripts/`: verificación de paquetes, reproducción de evaluaciones, publicación en Hugging Face, DagsHub y API pública.
- `tests/`: pruebas del ambiente, la API, los pipelines y las familias de cuadrícula.
- `docs/`: documentación técnica.
- `env/`, `agent/`, `train.py`, `visualize.py`, `plot3d.py`: adaptadores y herramientas heredadas.

La simulación Python es la autoridad para estados y transiciones. La interfaz Three.js solo representa el estado comunicado por la API; la velocidad de animación no modifica el paso físico.

## Experimentos y estados

Cada perfil declara su hipótesis antes de entrenar. Hay 19 perfiles y configuraciones con hipótesis, y todos los runs del registro local terminaron (`FINISHED`); los candidatos rechazados se conservan y su decisión queda en el pipeline y en [docs/decision.md](docs/decision.md). MLflow: `mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db` (con la base obtenida por `dvc pull`) o la interfaz de DagsHub.

## Límites conocidos

Un solo PPO completa N1 y solo con 1 de 3 semillas; N2 y N3 no se resuelven; no hay un test final reservado; la API depende de un equipo del grupo. La lista completa está en la sección 7 del [informe](docs/informe.md).

## Assets y licencias

La escena 3D se genera en código: el repositorio no incluye imágenes, texturas ni modelos 3D de terceros. `web/src/style.css` carga las tipografías DM Sans, DM Mono y Manrope desde Google Fonts (licencia SIL Open Font License). Dependencias de la web: three.js (MIT), Vite (MIT) y TypeScript (Apache-2.0). El repositorio es material académico y no declara una licencia de código abierto.
