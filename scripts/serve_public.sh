#!/bin/sh
# Publica la API local por HTTPS con ngrok. Solo para demostraciones: funciona mientras este equipo esté
# encendido, y cualquiera con la URL puede usar la API, incluido lanzar entrenamientos con el límite indicado.
# Uso: ./scripts/serve_public.sh   (requiere: ngrok config add-authtoken <token>)
set -eu
cd "$(dirname "$0")/.."
export MLFLOW_TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1
export DRONE_RESCUE_CORS_ORIGINS="${DRONE_RESCUE_CORS_ORIGINS:-https://drone-rescue-snowy.vercel.app,https://drone-rescue-visualizacion1.vercel.app,https://drone-rescue-git-feature-dqn-ambiente-continuo-visualizacion1.vercel.app}"
export DRONE_RESCUE_TRAINING_MAX_TIMESTEPS="${DRONE_RESCUE_TRAINING_MAX_TIMESTEPS:-100000}"
.venv/bin/python -m api.main &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null' EXIT
ngrok http 8000 --log=stdout
