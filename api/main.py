"""FastAPI entry point; the Python environment remains authoritative."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal
from urllib.parse import urlparse

import mlflow
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field

from agents.dqn import FEATURES_VERSION, make_features
from agents.ppo import PPOPolicy, input_contract, make_env
from envs.drone_env import DEFAULT_CONFIG_PATH, ENVIRONMENT_VERSION, DroneEnv, load_config
from pipelines.evaluate import (
    QLearningPolicy,
    RulePolicy,
    evaluate,
    load_q_table,
    load_suite,
)

from pipelines.tracking import configure_tracking
from pipelines.train_ppo import evaluate_model as evaluate_ppo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PROFILE_DIR = PROJECT_ROOT / "configs" / "training"
TRAINING_MAX_TIMESTEPS = int(os.getenv("DRONE_RESCUE_TRAINING_MAX_TIMESTEPS", "300000"))
EXPLORATION_EPSILON = 0.1  # modo entrenamiento con Q-table: acción aleatoria con esta probabilidad
MANIFEST_DIR = Path(os.getenv("DRONE_RESCUE_MANIFEST_DIR", PROJECT_ROOT / "manifests")).resolve()
EVALUATION_SUITE_PATH = Path(
    os.getenv("DRONE_RESCUE_EVALUATION_SUITE", PROJECT_ROOT / "configs" / "finalist_evaluation_suite.yaml")
).resolve()
API_VERSION = os.getenv("DRONE_RESCUE_API_VERSION", "0.1.0")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL")
EpisodeMode = Literal["manual", "aleatorio", "referencia", "evaluacion", "entrenamiento", "replay"]


class LoadRequest(BaseModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    seed: int = 0


class ResetRequest(BaseModel):
    mode: EpisodeMode = "manual"
    seed: int | None = None


class TrainingRequest(BaseModel):
    profile: str = Field(pattern=r"^ppo_[A-Za-z0-9_-]+$")
    seed: int = 0
    timesteps: int = Field(default=50_000, gt=0)


@dataclass
class ActiveRun:
    run_id: str
    manifest: dict
    model_path: Path
    environment_config: dict
    environment: DroneEnv
    q_table: dict
    temporary_directory: tempfile.TemporaryDirectory
    ppo_policy: PPOPolicy | None = None
    stabilized_env: object | None = None  # ambiente de la familia del modelo; comparte estado con environment
    model_family: str = "drone-rescue-v2"
    mode: str = "manual"
    observation: np.ndarray | None = None
    reward: float = 0.0
    episode_return: float = 0.0
    cause: str | None = None
    terminated: bool = False
    truncated: bool = False


class ApiRuntime:
    def __init__(self) -> None:
        self.active: ActiveRun | None = None
        self.lock = RLock()
        self.training: dict | None = None

    def require_active(self) -> ActiveRun:
        if self.active is None:
            raise HTTPException(status_code=409, detail="Primero carga un modelo con POST /load")
        return self.active


runtime = ApiRuntime()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(run_id: str) -> dict:
    path = MANIFEST_DIR / f"{run_id}.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"No existe el manifiesto del run {run_id}") from error
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail=f"El manifiesto de {run_id} no es JSON válido") from error
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=500, detail=f"El manifiesto de {run_id} debe ser un objeto JSON")
    return manifest


def _verified_artifact(manifest: dict, uri_field: str, hash_field: str, directory: Path) -> Path:
    uri = manifest.get(uri_field)
    expected_hash = manifest.get(hash_field)
    if not uri or not expected_hash:
        raise HTTPException(status_code=422, detail=f"El manifiesto no contiene {uri_field} y {hash_field}")
    try:
        downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
    except (MlflowException, FileNotFoundError, OSError) as error:
        raise HTTPException(status_code=502, detail=f"No se pudo descargar {uri_field} desde MLflow: {error}") from error
    if not downloaded.is_file():
        raise HTTPException(status_code=502, detail=f"MLflow no devolvió un archivo para {uri_field}")
    actual_hash = _sha256(downloaded)
    if actual_hash != expected_hash:
        raise HTTPException(status_code=409, detail=f"SHA-256 de {uri_field} no coincide con el manifiesto")
    target = directory / downloaded.name
    shutil.copyfile(downloaded, target)
    return target


def _validate_ppo_compatibility(manifest: dict, normalization: dict) -> None:
    if manifest.get("environment_version") != ENVIRONMENT_VERSION:
        raise HTTPException(status_code=409, detail="El modelo requiere otra environment_version")
    family = manifest.get("model_family")
    if family not in ("drone-rescue-v2", "drone-rescue-v3"):
        raise HTTPException(status_code=409, detail=f"Familia de modelo no soportada: {family!r}")
    if normalization != input_contract(family):
        raise HTTPException(status_code=409, detail="El contrato de entrada/salida no coincide con esta API")
    if normalization.get("features_version") != FEATURES_VERSION:
        raise HTTPException(status_code=409, detail="El escalado de entradas del modelo no coincide")


def _suite_for_manifest(manifest: dict) -> tuple[dict, Path]:
    """La suite con la que se registró el modelo, verificada por id, versión y SHA-256."""
    candidates = [EVALUATION_SUITE_PATH, *sorted((PROJECT_ROOT / "configs").glob("*evaluation_suite.yaml"))]
    for candidate in candidates:
        try:
            suite, path = load_suite(candidate)
        except (OSError, ValueError):
            continue
        if (
            suite["evaluation_suite_id"] == manifest.get("evaluation_suite_id")
            and suite["version"] == manifest.get("evaluation_suite_version")
            and _sha256(path) == manifest.get("evaluation_suite_sha256")
        ):
            return suite, path
    raise HTTPException(status_code=409, detail="Ninguna suite disponible coincide con la del manifiesto")


def _training_state() -> dict:
    job = runtime.training
    profiles = sorted(path.stem for path in TRAINING_PROFILE_DIR.glob("ppo_*.yaml"))
    base = {"profiles": profiles, "max_timesteps": TRAINING_MAX_TIMESTEPS, "state": "idle"}
    if job is None:
        return base
    code = job["process"].poll()
    log_text = Path(job["log"]).read_text(encoding="utf-8", errors="replace") if Path(job["log"]).exists() else ""
    progress = {}
    if Path(job["status"]).exists():
        try:
            progress = json.loads(Path(job["status"]).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            progress = {}
    run_id = None
    for line in reversed(log_text.splitlines()):
        if line.startswith('{"run_id"'):
            run_id = json.loads(line)["run_id"]
            break
    state = {
        **base,
        "state": "running" if code is None else ("finished" if code == 0 else "failed"),
        "profile": job["profile"],
        "seed": job["seed"],
        "timesteps": job["timesteps"],
        "elapsed_s": round((job.get("ended") or time.time()) - job["started"]),
        "progress": progress,
        "run_id": run_id,
    }
    if code not in (None, 0):
        state["error"] = log_text[-1500:]
    return state


def _validate_model_compatibility(manifest: dict, environment_config: dict, normalization: dict) -> None:
    if manifest.get("environment_version") != ENVIRONMENT_VERSION:
        raise HTTPException(
            status_code=409,
            detail=(
                f"El modelo requiere environment_version={manifest.get('environment_version')!r}; "
                f"esta API sirve {ENVIRONMENT_VERSION!r} y no tiene un adaptador histórico validado"
            ),
        )
    experiment = environment_config.get("experiment", {})
    level = environment_config.get("level", {})
    if not isinstance(experiment, dict) or not isinstance(level, dict):
        raise HTTPException(status_code=422, detail="Metadatos del ambiente inválidos")
    model_family = experiment.get("model_family") or level.get("model_family")
    if normalization.get("family") != model_family or model_family != "drone-rescue-v2":
        raise HTTPException(status_code=409, detail="La familia del modelo y la normalización no son compatibles")
    if normalization.get("method") != "discretize_observation":
        raise HTTPException(status_code=409, detail="Método de normalización no soportado por esta API")
    q_learning = environment_config.get("q_learning")
    parameters = q_learning.get("discretization") if isinstance(q_learning, dict) else None
    if not isinstance(parameters, dict) or normalization.get("parameters") != parameters:
        raise HTTPException(status_code=409, detail="La discretización no coincide con el ambiente del manifiesto")
    if normalization.get("state_dimensions") != 16 or normalization.get("action_dimensions") != 7:
        raise HTTPException(status_code=409, detail="Las dimensiones del modelo no corresponden a drone-rescue-v2")


def _supported_mode(mode: str) -> None:
    if mode == "replay":
        raise HTTPException(
            status_code=501,
            detail=f"El modo {mode!r} todavía no está implementado; no se devolverá una simulación sustituta",
        )


def _episode_payload(active: ActiveRun) -> dict:
    environment = active.environment
    return {
        "run_id": active.run_id,
        "mode": active.mode,
        "position_m": environment.position.astype(float).tolist(),
        "telemetry": {
            "velocity_mps": environment.velocity.astype(float).tolist(),
            "battery_wh": float(environment.battery_wh),
            "battery_fraction": float(environment.battery_wh / environment.config["battery"]["capacity_wh"]),
            "has_supply": bool(environment.has_supply),
            "stage": environment.stage,
            "contacts": int(environment.contacts),
            "steps": int(environment.steps),
            "wind_mps2": environment.wind.astype(float).tolist(),
            "environment_version": environment.environment_version,
        },
        "model_family": active.model_family,
        "reward": float(active.reward),
        "episode_return": float(active.episode_return),
        "cause": active.cause,
        "terminated": active.terminated,
        "truncated": active.truncated,
    }


def _catalog_manifest_map() -> dict[str, dict]:
    manifests = {}
    if not MANIFEST_DIR.is_dir():
        raise HTTPException(status_code=503, detail=f"No existe el directorio de manifiestos: {MANIFEST_DIR}")
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        if path.name == "model-manifest.template.json":
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=500, detail=f"El manifiesto {path.name} no es JSON válido") from error
        if not isinstance(manifest, dict):
            raise HTTPException(status_code=500, detail=f"El manifiesto {path.name} debe ser un objeto JSON")
        run_id = manifest.get("run_id")
        if run_id:
            manifests[str(run_id)] = manifest
    return manifests


def _mlflow_link(tracking_uri: str, experiment_id: str, run_id: str) -> str | None:
    base_url = MLFLOW_UI_URL or (
        tracking_uri if urlparse(tracking_uri).scheme in {"http", "https"} else None
    )
    if base_url:
        return f"{base_url.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
    return None


def _configuration_changes(baseline: dict, resolved: dict) -> list[dict]:
    changes = []

    def compare(base_value, run_value, prefix: str) -> None:
        if isinstance(base_value, dict) and isinstance(run_value, dict):
            for key in sorted(base_value.keys() | run_value.keys()):
                if not prefix and key in {"experiment", "level"}:
                    continue
                child_path = f"{prefix}.{key}" if prefix else key
                if key not in base_value:
                    changes.append({"path": child_path, "baseline": None, "run": run_value[key]})
                elif key not in run_value:
                    changes.append({"path": child_path, "baseline": base_value[key], "run": None})
                else:
                    compare(base_value[key], run_value[key], child_path)
        elif base_value != run_value:
            changes.append({"path": prefix, "baseline": base_value, "run": run_value})

    compare(baseline, resolved, "")
    return changes


def _catalog_artifacts(run_id: str, manifest: dict, baseline_config: dict) -> dict:
    details = {
        "changed_dimension": None,
        "changes_vs_baseline": None,
        "mean_steps": None,
        "config_error": None,
        "evaluation_error": None,
    }
    config_uri = manifest.get("config_uri")
    if config_uri:
        try:
            config_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=config_uri))
            if manifest.get("config_sha256") and _sha256(config_path) != manifest["config_sha256"]:
                raise ValueError("SHA-256 de config.json no coincide con el manifiesto")
            config_document = json.loads(config_path.read_text(encoding="utf-8"))
            environment_config = config_document.get("environment")
            if not isinstance(environment_config, dict):
                raise ValueError("config.json no contiene el ambiente resuelto")
            details["changes_vs_baseline"] = _configuration_changes(baseline_config, environment_config)
            experiment = environment_config.get("experiment", {})
            if isinstance(experiment, dict):
                details["changed_dimension"] = experiment.get("changed_dimension")
        except (MlflowException, FileNotFoundError, OSError, json.JSONDecodeError, ValueError, AttributeError) as error:
            details["config_error"] = f"No se pudo validar la configuración de MLflow: {error}"

    if manifest.get("model_sha256"):
        try:
            evaluation_uri = manifest.get(
                "evaluation_uri",
                f"runs:/{run_id}/evaluation/evaluacion.json",
            )
            evaluation_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=evaluation_uri))
            if manifest.get("evaluation_sha256") and _sha256(evaluation_path) != manifest["evaluation_sha256"]:
                raise ValueError("SHA-256 de evaluacion.json no coincide con el manifiesto")
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            case_steps = [
                case.get("metrics", {}).get("mean_steps")
                for case in evaluation.get("cases", [])
                if isinstance(case.get("metrics", {}).get("mean_steps"), (int, float))
            ]
            if case_steps:
                details["mean_steps"] = float(sum(case_steps) / len(case_steps))
        except (MlflowException, FileNotFoundError, OSError, json.JSONDecodeError, ValueError, AttributeError) as error:
            details["evaluation_error"] = f"No se pudo leer la evaluación registrada en MLflow: {error}"
    return details


def _optional_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _create_app(*, production: bool | None = None, cors_origins: list[str] | None = None) -> FastAPI:
    production = (
        os.getenv("DRONE_RESCUE_ENV", "development").lower() == "production"
        if production is None
        else production
    )
    if cors_origins is None:
        configured_origins = os.getenv("DRONE_RESCUE_CORS_ORIGINS")
        cors_origins = (
            [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
            if configured_origins is not None
            else (
                []
                if production
                else ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
            )
        )
    if production and any(urlparse(origin).scheme != "https" for origin in cors_origins):
        raise RuntimeError("En producción, todos los orígenes CORS deben usar HTTPS")
    if MLFLOW_UI_URL:
        ui_scheme = urlparse(MLFLOW_UI_URL).scheme
        if ui_scheme not in {"http", "https"} or (production and ui_scheme != "https"):
            raise RuntimeError("MLFLOW_UI_URL debe ser HTTPS en producción")

    application = FastAPI(title="Drone Rescue API", version=API_VERSION)
    if cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @application.middleware("http")
    async def enforce_https(request: Request, call_next):
        if production and request.url.scheme != "https":
            return JSONResponse(
                status_code=426,
                content={"detail": "HTTPS es obligatorio en producción"},
            )
        return await call_next(request)

    @application.get("/health")
    def health():
        with runtime.lock:
            return {"status": "ok", "model_loaded": runtime.active is not None}

    @application.get("/version")
    def version_info():
        commit = os.getenv("DRONE_RESCUE_GIT_COMMIT")
        if not commit:
            try:
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                commit = None
        with runtime.lock:
            active = runtime.active
            served = None if active is None else {
                "run_id": active.run_id,
                "model_sha256": active.manifest.get("model_sha256"),
                "registry_uri": active.manifest.get("registry_uri"),
                "model_family": active.model_family,
                "level": active.manifest.get("level_id"),
            }
        return {"api_version": API_VERSION, "environment_version": ENVIRONMENT_VERSION,
                "git_commit": commit, "served_model": served}

    @application.get("/training")
    def training_status():
        with runtime.lock:
            return _training_state()

    @application.post("/training")
    def start_training(request: TrainingRequest):
        if production and os.getenv("DRONE_RESCUE_ALLOW_TRAINING") != "1":
            raise HTTPException(status_code=403, detail="El entrenamiento está desactivado en este despliegue")
        if request.timesteps > TRAINING_MAX_TIMESTEPS:
            raise HTTPException(status_code=422, detail=f"El presupuesto máximo es {TRAINING_MAX_TIMESTEPS} pasos")
        profile = TRAINING_PROFILE_DIR / f"{request.profile}.yaml"
        if not profile.is_file():
            raise HTTPException(status_code=404, detail=f"No existe el perfil {request.profile}")
        with runtime.lock:
            if runtime.training and runtime.training["process"].poll() is None:
                raise HTTPException(status_code=409, detail="Ya hay un entrenamiento en curso")
            directory = tempfile.mkdtemp(prefix="drone-rescue-training-")
            log_path, status_path = Path(directory) / "train.log", Path(directory) / "status.json"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "pipelines.train_ppo", "--profile", str(profile), "--seed", str(request.seed),
                     "--timesteps", str(request.timesteps), "--status-file", str(status_path),
                     "--manifest-dir", str(MANIFEST_DIR)],
                    cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
            runtime.training = {"process": process, "log": str(log_path), "status": str(status_path),
                                "profile": request.profile, "seed": request.seed,
                                "timesteps": request.timesteps, "started": time.time()}
            return _training_state()

    @application.get("/catalog")
    def catalog():
        manifests = _catalog_manifest_map()
        try:
            baseline_config = load_config(DEFAULT_CONFIG_PATH)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=503, detail=f"No se pudo cargar el baseline: {error}") from error
        client = MlflowClient()
        try:
            experiments = client.search_experiments()
            experiment_names = {experiment.experiment_id: experiment.name for experiment in experiments}
            tracking_uri = mlflow.get_tracking_uri()
            entries = []
            for experiment in experiments:
                token = None
                while True:
                    page = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["attributes.start_time DESC"],
                        max_results=1000,
                        page_token=token,
                    )
                    for run in page:
                        run_id = run.info.run_id
                        manifest = manifests.get(run_id, {})
                        tags = run.data.tags
                        params = run.data.params
                        artifact_details = _catalog_artifacts(run_id, manifest, baseline_config)
                        duration_seconds = None
                        if run.info.start_time is not None and run.info.end_time is not None:
                            duration_seconds = max(0.0, (run.info.end_time - run.info.start_time) / 1000.0)
                        entries.append({
                            "run_id": run_id,
                            "status": run.info.status,
                            "checkpoint_status": tags.get("checkpoint_status"),
                            "hypothesis": tags.get("hypothesis"),
                            "commit": tags.get("git_commit") or manifest.get("git_commit"),
                            "level": tags.get("level_id") or manifest.get("level_id"),
                            "experiment_id": params.get("experiment_id") or tags.get("level_id") or manifest.get("level_id"),
                            "mlflow_experiment": experiment_names.get(run.info.experiment_id),
                            "environment_version": tags.get("environment_version") or manifest.get("environment_version"),
                            "seed": manifest.get("seed", _optional_int(params.get("seed"))),
                            "training_episodes": _optional_int(params.get("episodes")),
                            "training_timesteps": _optional_int(params.get("timesteps")),
                            "algorithm": tags.get("algorithm") or manifest.get("algorithm") or "q_learning",
                            "model_family": tags.get("model_family") or manifest.get("model_family") or "drone-rescue-v2",
                            "mean_steps": artifact_details["mean_steps"],
                            "duration_seconds": duration_seconds,
                            "model_version": manifest.get("registry_version"),
                            "model_available": bool(manifest.get("model_uri") and manifest.get("model_sha256")),
                            "changed_dimension": artifact_details["changed_dimension"],
                            "changes_vs_baseline": artifact_details["changes_vs_baseline"],
                            "config_error": artifact_details["config_error"],
                            "evaluation_error": artifact_details["evaluation_error"],
                            "metrics": dict(run.data.metrics),
                            "experiment": experiment_names.get(run.info.experiment_id),
                            "manifest_available": bool(manifest),
                            "model_sha256": manifest.get("model_sha256"),
                            "registry_uri": manifest.get("registry_uri"),
                            "mlflow_link": _mlflow_link(tracking_uri, run.info.experiment_id, run_id),
                            "mlflow_tracking_uri": tracking_uri,
                            "mlflow_run_uri": f"runs:/{run_id}",
                        })
                    token = page.token
                    if not token:
                        break
        except MlflowException as error:
            raise HTTPException(status_code=503, detail=f"No se pudo consultar MLflow: {error}") from error
        return {"runs": entries}

    @application.post("/load")
    def load_model(request: LoadRequest):
        manifest = _read_manifest(request.run_id)
        if str(manifest.get("run_id")) != request.run_id:
            raise HTTPException(status_code=422, detail="El run_id del manifiesto no coincide con la solicitud")
        temporary_directory = tempfile.TemporaryDirectory(prefix=f"drone-rescue-{request.run_id}-")
        directory = Path(temporary_directory.name)
        try:
            model_path = _verified_artifact(manifest, "model_uri", "model_sha256", directory)
            config_path = _verified_artifact(manifest, "config_uri", "config_sha256", directory)
            normalization_path = _verified_artifact(
                manifest, "normalization_uri", "normalization_sha256", directory
            )
            config_document = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config_document, dict):
                raise HTTPException(status_code=422, detail="config.json debe ser un objeto JSON")
            environment_config = config_document.get("environment")
            if not isinstance(environment_config, dict):
                raise HTTPException(status_code=422, detail="config.json no contiene el ambiente resuelto")
            normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
            if not isinstance(normalization, dict):
                raise HTTPException(status_code=422, detail="normalization.json debe ser un objeto JSON")
            is_ppo = manifest.get("algorithm") == "ppo"
            if is_ppo:
                _validate_ppo_compatibility(manifest, normalization)
            else:
                _validate_model_compatibility(manifest, environment_config, normalization)
            environment_config_path = directory / "environment.yaml"
            environment_config_path.write_text(
                yaml.safe_dump(environment_config, sort_keys=True),
                encoding="utf-8",
            )
            family = str(manifest.get("model_family", "drone-rescue-v2"))
            stabilized_env = make_env(environment_config_path, family) if is_ppo else None
            environment = stabilized_env.unwrapped if is_ppo else DroneEnv(config_path=environment_config_path)
            q_table = {} if is_ppo else load_q_table(model_path)
            ppo_policy = PPOPolicy(model_path) if is_ppo else None
            if not is_ppo and environment.action_space.n != int(normalization["action_dimensions"]):
                raise HTTPException(status_code=409, detail="El espacio de acciones no coincide con la normalización")
            observation, _ = environment.reset(seed=request.seed)
            new_active = ActiveRun(
                run_id=request.run_id,
                manifest=manifest,
                model_path=model_path,
                environment_config=environment_config,
                environment=environment,
                q_table=q_table,
                temporary_directory=temporary_directory,
                observation=observation,
                ppo_policy=ppo_policy,
                stabilized_env=stabilized_env,
                model_family=family,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            temporary_directory.cleanup()
            raise HTTPException(status_code=422, detail=f"Artefactos incompatibles o inválidos: {error}") from error
        except HTTPException:
            temporary_directory.cleanup()
            raise

        with runtime.lock:
            previous = runtime.active
            runtime.active = new_active
            if previous is not None:
                previous.environment.close()
                previous.temporary_directory.cleanup()
        return {
            "model": {
                "run_id": new_active.run_id,
                "sha256": manifest["model_sha256"],
                "level": manifest.get("level_id"),
                "commit": manifest.get("git_commit"),
            },
            "environment": {
                "version": environment.environment_version,
                "reward_version": environment.reward_version,
                "level": manifest.get("level_id"),
                "model_family": new_active.model_family if new_active.ppo_policy else (
                    environment_config.get("experiment", {}).get("model_family")
                    or environment_config.get("level", {}).get("model_family")
                ),
                "algorithm": "ppo" if new_active.ppo_policy else "q_learning",
            },
            "config": environment_config,
            "config_sha256": manifest["config_sha256"],
            "mode": new_active.mode,
        }

    @application.post("/evaluate")
    def evaluate_active_model():
        evaluation_directory = tempfile.TemporaryDirectory(prefix="drone-rescue-evaluation-")
        try:
            with runtime.lock:
                active = runtime.require_active()
                manifest = active.manifest.copy()
                model_path = Path(evaluation_directory.name) / active.model_path.name
                shutil.copyfile(active.model_path, model_path)
            _, suite_path = _suite_for_manifest(manifest)
            evaluation_id = str(uuid.uuid4())
            output_path = PROJECT_ROOT / "artifacts" / "results" / f"api-evaluation-{evaluation_id}.json"
            try:
                if manifest.get("algorithm") == "ppo":
                    result = evaluate_ppo(suite_path, model_path, manifest["model_family"], output_path)
                else:
                    result = evaluate(suite_path, model_path, output_path)
            except (ValueError, FileExistsError) as error:
                raise HTTPException(status_code=422, detail=f"No se pudo ejecutar la evaluación: {error}") from error
            return {
                "evaluation_id": evaluation_id,
                "run_id": manifest["run_id"],
                "result_path": str(output_path.relative_to(PROJECT_ROOT)),
                "result": result,
            }
        finally:
            evaluation_directory.cleanup()

    @application.post("/episode/reset")
    def reset_episode(request: ResetRequest):
        _supported_mode(request.mode)
        with runtime.lock:
            active = runtime.require_active()
            active.mode = request.mode
            if active.stabilized_env is not None:
                active.stabilized_env.reset(seed=request.seed)  # también reinicia el estabilizador
                active.observation = active.environment._observation()
            else:
                active.observation, _ = active.environment.reset(seed=request.seed)
            active.reward = 0.0
            active.episode_return = 0.0
            active.cause = None
            active.terminated = False
            active.truncated = False
            return _episode_payload(active)

    @application.get("/episode/step")
    def episode_step(
        mode: EpisodeMode | None = Query(default=None),
        action: int | None = Query(default=None),
    ):
        with runtime.lock:
            active = runtime.require_active()
            if mode is not None and mode != active.mode:
                raise HTTPException(status_code=409, detail="Cambia el modo con POST /episode/reset antes del paso")
            _supported_mode(active.mode)
            if active.terminated or active.truncated:
                raise HTTPException(status_code=409, detail=f"El episodio terminó con causa {active.cause!r}; reinícialo")
            observation = active.observation
            if observation is None:
                raise HTTPException(status_code=409, detail="El episodio no tiene una observación activa")
            if active.mode == "manual":
                if action is None:
                    raise HTTPException(status_code=422, detail="El modo manual requiere el parámetro action")
                selected_action = action
            elif active.mode == "aleatorio":
                if action is not None:
                    raise HTTPException(status_code=422, detail="El modo aleatorio no acepta action")
                selected_action = int(active.environment.np_random.integers(active.environment.action_space.n))
            elif active.mode == "referencia":
                if action is not None:
                    raise HTTPException(status_code=422, detail="El modo referencia no acepta action")
                selected_action = RulePolicy().act(observation, active.environment.config)
            else:
                if action is not None:
                    raise HTTPException(status_code=422, detail=f"El modo {active.mode!r} no acepta action")
                # evaluación: modelo congelado sin exploración; entrenamiento: misma política explorando
                explore = active.mode == "entrenamiento"
                if active.ppo_policy is not None:
                    features = make_features(observation)
                    model_action = int(active.ppo_policy.model.predict(features, deterministic=not explore)[0])
                    _, reward, terminated, truncated, info = active.stabilized_env.step(model_action)
                    return _apply_transition(active, active.environment._observation(), reward, terminated, truncated, info)
                if explore and active.environment.np_random.random() < EXPLORATION_EPSILON:
                    selected_action = int(active.environment.np_random.integers(active.environment.action_space.n))
                else:
                    selected_action = QLearningPolicy(active.q_table).act(observation, active.environment.config)
            if not active.environment.action_space.contains(selected_action):
                raise HTTPException(status_code=422, detail=f"Acción fuera del espacio válido: {selected_action}")
            next_observation, reward, terminated, truncated, info = active.environment.step(selected_action)
            return _apply_transition(active, next_observation, reward, terminated, truncated, info)

    return application


def _apply_transition(active: ActiveRun, observation, reward, terminated, truncated, info) -> dict:
    active.observation = observation
    active.reward = float(reward)
    active.episode_return += float(reward)
    active.cause = info["cause"]
    active.terminated = bool(terminated)
    active.truncated = bool(truncated)
    return _episode_payload(active)


app = _create_app()


def main() -> None:
    os.chdir(PROJECT_ROOT)
    configure_tracking()
    production = os.getenv("DRONE_RESCUE_ENV", "development").lower() == "production"
    options = {
        "app": "api.main:app",
        "host": os.getenv("DRONE_RESCUE_API_HOST", "127.0.0.1"),
        "port": int(os.getenv("DRONE_RESCUE_API_PORT", "8000")),
    }
    if production:
        certificate = os.getenv("DRONE_RESCUE_SSL_CERTFILE")
        private_key = os.getenv("DRONE_RESCUE_SSL_KEYFILE")
        if not certificate or not private_key:
            raise RuntimeError("Producción requiere DRONE_RESCUE_SSL_CERTFILE y DRONE_RESCUE_SSL_KEYFILE")
        options.update(ssl_certfile=certificate, ssl_keyfile=private_key)
    uvicorn.run(**options)


if __name__ == "__main__":
    main()
