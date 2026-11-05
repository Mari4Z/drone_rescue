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
- Node.js LTS será necesario cuando se implemente la aplicación web; aún no se
  necesita para entrenar el agente.

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

## Arquitectura

- `envs/`: entorno Gymnasium canónico y reglas físicas de la simulación.
- `agents/`: agente Q-learning tabular.
- `env/` y `agent/`: adaptadores para preservar imports históricos.
- `pipelines/train.py`: entrenamiento MLflow, evaluación greedy, checkpoint,
  Model Registry y manifiesto; `train.py` conserva la ruta CLI anterior.
- `visualize.py`, `plot3d.py`: vistas de depuración y visualización estática.
- `configs/`: configuración YAML efectiva del ambiente y recompensa.
- `pipelines/`: entrenamiento y evaluación; Prefect todavía está pendiente.
- `tests/`: pruebas del contrato actual del entorno e imports.
- `web/`: reservado para Vite, TypeScript, Three.js y Vercel.
- `manifests/`: manifiestos por run con hashes, URIs y runtime.
- `artifacts/`: destinos DVC para Q-tables, suites y resultados.
- `docs/`: auditoría y documentación técnica.

La simulación Python es la autoridad para estados y transiciones. La futura
interfaz Three.js solo deberá representar el estado comunicado por la API; la
velocidad de animación no modificará el paso físico.

## Pruebas

```bash
pytest
```

## Estado frente al producto final

La auditoría en [`docs/auditoria.md`](docs/auditoria.md) registra lo que existe,
lo preparado y lo que falta. No se encontró el PDF de requisitos en la rama
revisada, así que falta contrastar este inventario con el documento oficial.
