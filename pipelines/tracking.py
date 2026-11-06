"""Ubicación versionada del registro MLflow del proyecto.

Base de datos y artefactos viven en artifacts/mlflow, seguidos por DVC. Los artefactos usan una ruta
relativa para que el registro siga funcionando al recuperarlo en otro equipo, desde la raíz del repo.
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
