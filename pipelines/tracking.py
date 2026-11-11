"""Ubicación versionada del registro MLflow del proyecto.

Base de datos y artefactos viven en artifacts/mlflow, seguidos por DVC. MLflow guarda rutas absolutas,
así que este registro solo resuelve artefactos en el equipo donde se creó; en otro equipo el modelo se
recupera con el paquete de artifacts/models y se verifica con los SHA-256 del manifiesto.
"""

from __future__ import annotations

import os

import mlflow

DEFAULT_TRACKING_URI = "sqlite:///artifacts/mlflow/mlflow.db"
ARTIFACT_ROOT = "artifacts/mlflow/mlruns"
EXPERIMENT = "drone-rescue-training"


def configure_tracking(experiment: str = EXPERIMENT) -> str:
    os.makedirs("artifacts/mlflow", exist_ok=True)
    uri = os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=f"{ARTIFACT_ROOT}/{experiment}")
    return uri
