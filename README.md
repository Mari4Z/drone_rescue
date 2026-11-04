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
python train.py
```

El entrenamiento usa por defecto 20.000 episodios y semilla 0. Al terminar,
imprime el primer éxito y una ruta greedy y muestra vistas ASCII. Para producir
la imagen 3D estática:

```bash
python plot3d.py
```

Por defecto crea una ruta única dentro de `outputs/`. Si indicas `--out`, la
ruta debe ser nueva; el programa se niega a sobrescribir una imagen existente.
Para abrir la ventana interactiva de Matplotlib:

```bash
python plot3d.py --show
```

El entrenamiento actual no guarda una Q-table ni envía sus métricas a MLflow.
MLflow se incluye para registrar y reproducir cambios, pero falta instrumentar
`train.py` y habilitar el versionado y recuperación de modelos.

## Reproducir el entrenamiento actual

1. Usa el commit de código deseado y crea el entorno con `requirements.txt`.
2. Ejecuta `python train.py`; `train()` fija la semilla inicial en 0.
3. Conserva el commit, la salida de consola y cualquier artefacto generado como
   evidencia. La corrida actual no registra métricas de entrenamiento en
   MLflow; ese flujo aún debe implementarse para obtener un run reproducible
   desde `train.py`.

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
- `train.py`: bucle de entrenamiento y recorrido greedy.
- `visualize.py`, `plot3d.py`: vistas de depuración y visualización estática.
- `configs/`: configuración YAML efectiva del ambiente y recompensa.
- `pipelines/`: reservado para orquestación Prefect.
- `tests/`: pruebas del contrato actual del entorno e imports.
- `web/`: reservado para Vite, TypeScript, Three.js y Vercel.
- `manifests/`: plantilla de manifiesto para modelos y corridas.
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
