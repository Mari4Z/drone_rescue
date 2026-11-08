"""FastAPI entry point; the Python environment remains authoritative."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock, Thread
from typing import Literal
from urllib.parse import urlparse

import mlflow
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field

from envs.drone_env import DEFAULT_CONFIG_PATH, ENVIRONMENT_VERSION, DroneEnv, load_config
from pipelines.evaluate import (
    QLearningPolicy,
    RulePolicy,
    evaluate,
    load_q_table,
    load_suite,
)
from pipelines.train import load_profile, train_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = Path(os.getenv("DRONE_RESCUE_MANIFEST_DIR", PROJECT_ROOT / "manifests")).resolve()
EVALUATION_SUITE_PATH = Path(
    os.getenv("DRONE_RESCUE_EVALUATION_SUITE", PROJECT_ROOT / "configs" / "finalist_evaluation_suite.yaml")
).resolve()
API_VERSION = os.getenv("DRONE_RESCUE_API_VERSION", "0.1.0")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL")
EpisodeMode = Literal["manual", "aleatorio", "referencia", "evaluacion", "entrenamiento", "replay"]
CONFIG_PRESETS = {
    "baseline": DEFAULT_CONFIG_PATH,
    "n1": PROJECT_ROOT / "configs" / "levels" / "n1_mision_base.yaml",
    "n2": PROJECT_ROOT / "configs" / "levels" / "n2_mas_decisiones.yaml",
    "n3": PROJECT_ROOT / "configs" / "levels" / "n3_operacion_exigente.yaml",
}


class LoadRequest(BaseModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    seed: int = 0
    session_id: str | None = Field(default=None, pattern=r"^[0-9a-fA-F-]{36}$")


class ResetRequest(BaseModel):
    mode: EpisodeMode = "manual"
    seed: int | None = None


class ManualSessionRequest(BaseModel):
    config_id: Literal["baseline", "n1", "n2", "n3"] = "baseline"
    seed: int = 0


class TrainingRequest(BaseModel):
    profile_id: Literal["n1", "n2", "n3"]
    seed: int = 0
    episodes: int | None = Field(default=None, ge=1, le=15000)


@dataclass
class ActiveRun:
    session_id: str
    run_id: str | None
    manifest: dict
    model_path: Path | None
    environment_config: dict
    environment: DroneEnv
    q_table: dict | None
    temporary_directory: tempfile.TemporaryDirectory | None
    config_id: str
    mode: str = "manual"
    observation: np.ndarray | None = None
    reward: float = 0.0
    cause: str | None = None
    terminated: bool = False
    truncated: bool = False
    last_access: float = 0.0


class ApiRuntime:
    def __init__(self) -> None:
        self.sessions: dict[str, ActiveRun] = {}
        self.lock = RLock()
        self.training_jobs: dict[str, dict] = {}
        self.jobs_lock = RLock()

    def require_active(self, session_id: str | None) -> ActiveRun:
        if not session_id:
            raise HTTPException(status_code=400, detail="Falta el encabezado X-Drone-Session")
        active = self.sessions.get(session_id)
        if active is None:
            raise HTTPException(status_code=409, detail="La sesión no existe o expiró; inicia otra con POST /session o /load")
        active.last_access = time.monotonic()
        return active

    def replace_session(self, active: ActiveRun) -> None:
        now = time.monotonic()
        expired = [
            session_id
            for session_id, session in self.sessions.items()
            if now - session.last_access > 1800
        ]
        for session_id in expired:
            self.close_session(session_id)
        if active.session_id not in self.sessions and len(self.sessions) >= 32:
            active.environment.close()
            if active.temporary_directory is not None:
                active.temporary_directory.cleanup()
            raise HTTPException(status_code=429, detail="Se alcanzó el límite de 32 sesiones activas")
        previous = self.sessions.get(active.session_id)
        if previous is not None:
            previous.environment.close()
            if previous.temporary_directory is not None:
                previous.temporary_directory.cleanup()
        active.last_access = now
        self.sessions[active.session_id] = active

    def close_session(self, session_id: str) -> None:
        previous = self.sessions.pop(session_id, None)
        if previous is not None:
            previous.environment.close()
            if previous.temporary_directory is not None:
                previous.temporary_directory.cleanup()


runtime = ApiRuntime()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(run_id: str) -> dict:
    tracking_uri = mlflow.get_tracking_uri()
    if tracking_uri.startswith(("http://", "https://")):
        remote_run = _find_imported_run(run_id)
        if remote_run is not None:
            return _download_manifest(remote_run.info.run_id, run_id)
    path = MANIFEST_DIR / f"{run_id}.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        try:
            artifact = Path(mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=f"manifest/{run_id}.json",
            ))
            manifest = json.loads(artifact.read_text(encoding="utf-8"))
        except (MlflowException, FileNotFoundError, OSError, json.JSONDecodeError) as remote_error:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró el manifiesto del run {run_id} localmente ni en MLflow: {remote_error}",
            ) from error
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail=f"El manifiesto de {run_id} no es JSON válido") from error
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=500, detail=f"El manifiesto de {run_id} debe ser un objeto JSON")
    return manifest


def _find_imported_run(source_run_id: str):
    client = MlflowClient()
    for experiment in client.search_experiments():
        page = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                f"tags.source_run_id = '{source_run_id}' "
                "AND tags.migration_status = 'COMPLETED'"
            ),
            max_results=1,
        )
        if page:
            return page[0]
    return None


def _download_manifest(remote_run_id: str, source_run_id: str) -> dict:
    try:
        artifact = Path(mlflow.artifacts.download_artifacts(
            run_id=remote_run_id,
            artifact_path=f"manifest/{source_run_id}.json",
        ))
        manifest = json.loads(artifact.read_text(encoding="utf-8"))
    except (MlflowException, FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"No se pudo descargar el manifiesto remoto del run {source_run_id}: {error}",
        ) from error
    if not isinstance(manifest, dict) or str(manifest.get("run_id")) != source_run_id:
        raise HTTPException(status_code=502, detail=f"El manifiesto remoto del run {source_run_id} no es válido")
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
    if mode == "entrenamiento":
        raise HTTPException(status_code=409, detail="El modo entrenamiento inicia un trabajo con POST /training")


def _episode_payload(active: ActiveRun) -> dict:
    environment = active.environment
    return {
        "run_id": active.run_id or f"manual-{active.config_id}",
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
        "reward": float(active.reward),
        "cause": active.cause,
        "terminated": active.terminated,
        "truncated": active.truncated,
    }


def _training_worker(job_id: str, request: TrainingRequest) -> None:
    job = runtime.training_jobs[job_id]
    profile_path = PROJECT_ROOT / "configs" / "training" / f"{request.profile_id}.yaml"

    def update_progress(progress: dict) -> None:
        with runtime.jobs_lock:
            job.update(progress)

    try:
        result = train_run(
            profile_path,
            request.seed,
            episodes_override=request.episodes,
            progress_callback=update_progress,
        )
    except Exception as error:
        with runtime.jobs_lock:
            job.update({
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
            })
        return
    with runtime.jobs_lock:
        job.update({
            "status": "FINISHED",
            "run_id": result["run_id"],
            "manifest_path": result["manifest_path"],
            "checkpoint_path": result["checkpoint_path"],
        })


def _catalog_manifest_map() -> dict[str, dict]:
    manifests = {}
    if MANIFEST_DIR.is_dir():
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


def _create_app(
    *,
    production: bool | None = None,
    cors_origins: list[str] | None = None,
    api_token: str | None = None,
) -> FastAPI:
    production = (
        os.getenv("DRONE_RESCUE_ENV", "development").lower() == "production"
        if production is None
        else production
    )
    api_token = os.getenv("DRONE_RESCUE_API_TOKEN") if api_token is None else api_token
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
    if production and not api_token:
        raise RuntimeError("Producción requiere DRONE_RESCUE_API_TOKEN")
    if MLFLOW_UI_URL:
        ui_scheme = urlparse(MLFLOW_UI_URL).scheme
        if ui_scheme not in {"http", "https"} or (production and ui_scheme != "https"):
            raise RuntimeError("MLFLOW_UI_URL debe ser HTTPS en producción")

    application = FastAPI(title="Drone Rescue API", version=API_VERSION)

    @application.middleware("http")
    async def enforce_https(request: Request, call_next):
        forwarded_scheme = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
        if production and forwarded_scheme != "https":
            return JSONResponse(
                status_code=426,
                content={"detail": "HTTPS es obligatorio en producción"},
            )
        if (
            api_token
            and request.method != "OPTIONS"
            and request.url.path not in {"/health", "/version"}
            and not hmac.compare_digest(request.headers.get("x-api-key", ""), api_token)
        ):
            return JSONResponse(status_code=401, content={"detail": "Clave de API ausente o inválida"})
        return await call_next(request)

    @application.get("/health")
    def health():
        with runtime.lock:
            return {
                "status": "ok",
                "model_loaded": any(session.q_table is not None for session in runtime.sessions.values()),
            }

    @application.get("/version")
    def version_info():
        return {"api_version": API_VERSION, "environment_version": ENVIRONMENT_VERSION}

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
                        remote_run_id = run.info.run_id
                        tags = run.data.tags
                        if (
                            tags.get("source_run_id")
                            and tags.get("migration_status") != "COMPLETED"
                        ):
                            continue
                        run_id = tags.get("source_run_id", remote_run_id)
                        manifest = (
                            _read_manifest(run_id)
                            if tags.get("source_run_id")
                            else manifests.get(run_id, {})
                        )
                        if not manifest and tags.get("checkpoint_status") == "disponible":
                            try:
                                manifest = _read_manifest(run_id)
                            except HTTPException as error:
                                if error.status_code != 404:
                                    raise
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
                            "mlflow_link": _mlflow_link(tracking_uri, run.info.experiment_id, remote_run_id),
                            "mlflow_tracking_uri": tracking_uri,
                            "mlflow_run_uri": f"runs:/{remote_run_id}",
                        })
                    token = page.token
                    if not token:
                        break
        except MlflowException as error:
            raise HTTPException(status_code=503, detail=f"No se pudo consultar MLflow: {error}") from error
        return {"runs": entries}

    @application.post("/session")
    def create_manual_session(request: ManualSessionRequest):
        config_path = CONFIG_PRESETS[request.config_id]
        try:
            environment_config = load_config(config_path)
            environment = DroneEnv(config_path=config_path)
            observation, _ = environment.reset(seed=request.seed)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=f"No se pudo iniciar el ambiente: {error}") from error
        session_id = str(uuid.uuid4())
        new_active = ActiveRun(
            session_id=session_id,
            run_id=None,
            manifest={},
            model_path=None,
            environment_config=environment_config,
            environment=environment,
            q_table=None,
            temporary_directory=None,
            config_id=request.config_id,
            observation=observation,
        )
        with runtime.lock:
            runtime.replace_session(new_active)
        return {
            "session_id": session_id,
            "environment": {
                "version": environment.environment_version,
                "reward_version": environment.reward_version,
                "level": environment_config.get("level", {}).get("id", request.config_id),
                "model_family": (
                    environment_config.get("experiment", {}).get("model_family")
                    or environment_config.get("level", {}).get("model_family")
                ),
            },
            "config": environment_config,
            "mode": "manual",
            "state": _episode_payload(new_active),
        }

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
            _validate_model_compatibility(manifest, environment_config, normalization)
            environment_config_path = directory / "environment.yaml"
            environment_config_path.write_text(
                yaml.safe_dump(environment_config, sort_keys=True),
                encoding="utf-8",
            )
            environment = DroneEnv(config_path=environment_config_path)
            q_table = load_q_table(model_path)
            if environment.action_space.n != int(normalization["action_dimensions"]):
                raise HTTPException(status_code=409, detail="El espacio de acciones no coincide con la normalización")
            observation, _ = environment.reset(seed=request.seed)
            new_active = ActiveRun(
                session_id=request.session_id or str(uuid.uuid4()),
                run_id=request.run_id,
                manifest=manifest,
                model_path=model_path,
                environment_config=environment_config,
                environment=environment,
                q_table=q_table,
                temporary_directory=temporary_directory,
                config_id=str(manifest.get("level_id") or "model"),
                observation=observation,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            temporary_directory.cleanup()
            raise HTTPException(status_code=422, detail=f"Artefactos incompatibles o inválidos: {error}") from error
        except HTTPException:
            temporary_directory.cleanup()
            raise

        with runtime.lock:
            runtime.replace_session(new_active)
        return {
            "session_id": new_active.session_id,
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
                "model_family": (
                    environment_config.get("experiment", {}).get("model_family")
                    or environment_config.get("level", {}).get("model_family")
                ),
            },
            "config": environment_config,
            "config_sha256": manifest["config_sha256"],
            "mode": new_active.mode,
        }

    @application.post("/training", status_code=202)
    def start_training(request: TrainingRequest):
        with runtime.jobs_lock:
            if any(job["status"] == "RUNNING" for job in runtime.training_jobs.values()):
                raise HTTPException(status_code=409, detail="Ya hay un entrenamiento en curso")
            job_id = str(uuid.uuid4())
            profile_path = PROJECT_ROOT / "configs" / "training" / f"{request.profile_id}.yaml"
            if not profile_path.is_file():
                raise HTTPException(status_code=404, detail=f"No existe el perfil {request.profile_id}")
            try:
                profile, _ = load_profile(profile_path)
            except (OSError, ValueError) as error:
                raise HTTPException(status_code=422, detail=f"Perfil de entrenamiento inválido: {error}") from error
            total_episodes = request.episodes or int(profile["training"]["episodes"])
            job = {
                "job_id": job_id,
                "status": "RUNNING",
                "profile_id": request.profile_id,
                "seed": request.seed,
                "episode": 0,
                "total_episodes": total_episodes,
                "run_id": None,
                "error": None,
            }
            runtime.training_jobs[job_id] = job
        Thread(target=_training_worker, args=(job_id, request), daemon=True).start()
        return job

    @application.get("/training/{job_id}")
    def training_status(job_id: str):
        with runtime.jobs_lock:
            job = runtime.training_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"No existe el trabajo {job_id}")
            return job.copy()

    @application.post("/evaluate")
    def evaluate_active_model(session_id: str | None = Header(default=None, alias="X-Drone-Session")):
        evaluation_directory = tempfile.TemporaryDirectory(prefix="drone-rescue-evaluation-")
        try:
            with runtime.lock:
                active = runtime.require_active(session_id)
                if active.model_path is None or not active.manifest:
                    raise HTTPException(status_code=409, detail="La evaluación requiere un modelo cargado")
                manifest = active.manifest.copy()
                model_path = Path(evaluation_directory.name) / active.model_path.name
                shutil.copyfile(active.model_path, model_path)
            try:
                suite, suite_path = load_suite(EVALUATION_SUITE_PATH)
            except (OSError, ValueError) as error:
                raise HTTPException(status_code=422, detail=f"No se pudo cargar la suite: {error}") from error
            suite_sha256 = _sha256(suite_path)
            if (
                suite["evaluation_suite_id"] != manifest.get("evaluation_suite_id")
                or suite["version"] != manifest.get("evaluation_suite_version")
                or suite_sha256 != manifest.get("evaluation_suite_sha256")
            ):
                raise HTTPException(status_code=409, detail="La suite configurada no coincide con el manifiesto")
            evaluation_id = str(uuid.uuid4())
            output_path = Path(evaluation_directory.name) / "evaluacion.json"
            try:
                result = evaluate(suite_path, model_path, output_path)
            except (ValueError, FileExistsError) as error:
                raise HTTPException(status_code=422, detail=f"No se pudo ejecutar la evaluación: {error}") from error
            parent_run_id = str(manifest.get("remote_run_id") or manifest["run_id"])
            try:
                parent = MlflowClient().get_run(parent_run_id)
                with mlflow.start_run(
                    experiment_id=parent.info.experiment_id,
                    run_name=f"api-evaluation-{evaluation_id}",
                    tags={
                        "drone_rescue.evaluation_of": str(manifest["run_id"]),
                        "drone_rescue.evaluation_id": evaluation_id,
                    },
                ) as evaluation_run:
                    mlflow.log_param("evaluation_suite_id", suite["evaluation_suite_id"])
                    mlflow.log_param("evaluation_suite_version", suite["version"])
                    mlflow.log_artifact(str(output_path), artifact_path="evaluation")
                    for case in result.get("cases", []):
                        case_id = str(case["case_id"])
                        for metric, value in case.get("metrics", {}).items():
                            if isinstance(value, (int, float)) and np.isfinite(value):
                                mlflow.log_metric(f"{case_id}/{metric}", float(value))
                    evaluation_run_id = evaluation_run.info.run_id
            except (MlflowException, OSError, ValueError, KeyError) as error:
                raise HTTPException(status_code=502, detail=f"No se pudo guardar la evaluación en MLflow: {error}") from error
            return {
                "evaluation_id": evaluation_id,
                "run_id": manifest["run_id"],
                "evaluation_run_id": evaluation_run_id,
                "result_path": f"runs:/{evaluation_run_id}/evaluation/evaluacion.json",
                "result": result,
            }
        finally:
            evaluation_directory.cleanup()

    @application.post("/episode/reset")
    def reset_episode(
        request: ResetRequest,
        session_id: str | None = Header(default=None, alias="X-Drone-Session"),
    ):
        _supported_mode(request.mode)
        with runtime.lock:
            active = runtime.require_active(session_id)
            if request.mode in {"evaluacion", "replay"} and active.q_table is None:
                raise HTTPException(status_code=409, detail=f"El modo {request.mode!r} requiere un modelo cargado")
            active.mode = request.mode
            active.observation, _ = active.environment.reset(seed=request.seed)
            active.reward = 0.0
            active.cause = None
            active.terminated = False
            active.truncated = False
            return _episode_payload(active)

    @application.get("/episode/step")
    def episode_step(
        mode: EpisodeMode | None = Query(default=None),
        action: int | None = Query(default=None),
        session_id: str | None = Header(default=None, alias="X-Drone-Session"),
    ):
        with runtime.lock:
            active = runtime.require_active(session_id)
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
                if active.q_table is None:
                    raise HTTPException(status_code=409, detail=f"El modo {active.mode!r} requiere un modelo cargado")
                selected_action = QLearningPolicy(active.q_table).act(observation, active.environment.config)
            if not active.environment.action_space.contains(selected_action):
                raise HTTPException(status_code=422, detail=f"Acción fuera del espacio válido: {selected_action}")
            next_observation, reward, terminated, truncated, info = active.environment.step(selected_action)
            active.observation = next_observation
            active.reward = float(reward)
            active.cause = info["cause"]
            active.terminated = bool(terminated)
            active.truncated = bool(truncated)
            return _episode_payload(active)

    if cors_origins:
        return CORSMiddleware(
            application,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Drone-Session"],
        )
    return application


app = _create_app()


def main() -> None:
    production = os.getenv("DRONE_RESCUE_ENV", "development").lower() == "production"
    options = {
        "app": "api.main:app",
        "host": os.getenv("DRONE_RESCUE_API_HOST", "127.0.0.1"),
        "port": int(os.getenv("DRONE_RESCUE_API_PORT", "8000")),
    }
    if production:
        certificate = os.getenv("DRONE_RESCUE_SSL_CERTFILE")
        private_key = os.getenv("DRONE_RESCUE_SSL_KEYFILE")
        if certificate and private_key:
            options.update(ssl_certfile=certificate, ssl_keyfile=private_key)
        elif os.getenv("DRONE_RESCUE_TRUST_PROXY_HTTPS", "").lower() != "true":
            raise RuntimeError(
                "Producción requiere TLS directo o DRONE_RESCUE_TRUST_PROXY_HTTPS=true tras un proxy HTTPS confiable"
            )
    uvicorn.run(**options)


if __name__ == "__main__":
    main()
