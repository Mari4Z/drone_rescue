#!/usr/bin/env bash
# Conecta el proyecto a DagsHub: remoto DVC y MLflow compartido.
#
# Antes: inicia sesión en https://dagshub.com con GitHub, conecta el repositorio
# (Create → New Repository → Connect a repository → GitHub → drone_rescue) y crea
# un token en User Settings → Tokens.
#
# Uso, desde la raíz del proyecto y en tu propia terminal:
#   ./scripts/setup_dagshub.sh
#
# El token solo vive en esta terminal y en .dvc/config.local, que Git ignora.
set -euo pipefail
cd "$(dirname "$0")/.."

read -rp "Usuario de DagsHub: " DAGSHUB_USER
read -rp "Repositorio en DagsHub [drone_rescue]: " DAGSHUB_REPO
DAGSHUB_REPO=${DAGSHUB_REPO:-drone_rescue}
read -rsp "Token de DagsHub (no se muestra): " DAGSHUB_TOKEN
printf '\n'
trap 'unset DAGSHUB_TOKEN MLFLOW_TRACKING_PASSWORD' EXIT

BASE="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}"

echo "1/4 Compruebo las credenciales y el repositorio"
status=$(curl -s -o /dev/null -w '%{http_code}' -u "${DAGSHUB_USER}:${DAGSHUB_TOKEN}" \
  "https://dagshub.com/api/v1/repos/${DAGSHUB_USER}/${DAGSHUB_REPO}")
if [ "$status" != "200" ]; then
  echo "DagsHub respondió $status: revisa el usuario, el token y que el repositorio exista." >&2
  exit 1
fi

echo "2/4 Configuro el remoto DVC (la URL se versiona; el token queda solo en config.local)"
.venv/bin/dvc remote add -d -f dagshub "${BASE}.dvc"
.venv/bin/dvc remote modify --local dagshub auth basic
.venv/bin/dvc remote modify --local dagshub user "$DAGSHUB_USER"
.venv/bin/dvc remote modify --local dagshub password "$DAGSHUB_TOKEN"

echo "3/4 Subo modelos, tablas Q, resultados, suites y MLflow a DVC"
.venv/bin/dvc push -r dagshub
.venv/bin/dvc status -c -r dagshub

echo "4/4 Copio los runs de MLflow al servidor compartido"
export MLFLOW_TRACKING_USERNAME="$DAGSHUB_USER" MLFLOW_TRACKING_PASSWORD="$DAGSHUB_TOKEN"
export MLFLOW_DISABLE_AGENT_HINT=1
# d0c7d6e2…, a21f0fd5… y d5e6d157…: entrenamientos de prueba desde la web (11-nov), fuera a propósito.
# Si se interrumpe, vuelve a ejecutar el script: los runs ya copiados no se repiten.
.venv/bin/python scripts/migrate_mlflow_to_remote.py \
  --source-uri "sqlite:///${PWD}/artifacts/mlflow/mlflow.db" \
  --destination-uri "${BASE}.mlflow" \
  --skip-run d0c7d6e221cc4c2a81d8f37cd454a4d5 \
  --skip-run a21f0fd55c564bff8c05e78d558621d0 \
  --skip-run d5e6d157397242478274564bfbbe12bb

echo
echo "Listo. Experimentos: ${BASE}/experiments  ·  Datos DVC: ${BASE}"
