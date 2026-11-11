"""Etapas del pipeline de la sección 14 de la guía. Cada una deja evidencia y falla con diagnóstico.

Se ejecutan por separado para que el orquestador registre su estado:

    python -m pipelines.pipeline_steps validate --state S --profile P
    python -m pipelines.pipeline_steps tests
    python -m pipelines.pipeline_steps train --state S --profile P --seed 11 [--timesteps N | --run-id R]
    python -m pipelines.pipeline_steps evaluate --state S
    python -m pipelines.pipeline_steps register --state S
    python -m pipelines.pipeline_steps select --state S
    python -m pipelines.pipeline_steps publish --state S
    python -m pipelines.pipeline_steps verify --level N1
    python -m pipelines.pipeline_steps rollback --level N1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo import make_env
from envs.drone_env import ENVIRONMENT_VERSION
from pipelines.evaluate import DEFAULT_SUITE_PATH, load_suite
from pipelines.tracking import configure_tracking
from pipelines.train_ppo import DEFAULT_MANIFEST_DIR, _sha256, evaluate_model, load_ppo_profile, train_run

CRITERIA_PATH = PROJECT_ROOT / "configs" / "selection_criteria.yaml"
PUBLISHED_DIR = PROJECT_ROOT / "manifests" / "published"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
FAST_TESTS = ["tests/test_environment_contract.py", "tests/test_dqn.py"]
PRODUCTION_ALIAS = "produccion"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_state(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {"stages": {}}


def _write_state(path: str | Path, stage: str, evidence: dict, **fields) -> dict:
    state = _read_state(path)
    state.update(fields)
    state["stages"][stage] = {"finished_at": _now(), **evidence}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, **evidence}, ensure_ascii=False))
    return state


def _manifest(run_id: str, manifest_dir: Path = DEFAULT_MANIFEST_DIR) -> dict:
    return json.loads((Path(manifest_dir) / f"{run_id}.json").read_text(encoding="utf-8"))


def validate(state_path: str, profile_path: str) -> None:
    """Etapa 1: perfil, rutas, esquema y compatibilidad del ambiente."""
    profile, resolved = load_ppo_profile(profile_path)
    config = PROJECT_ROOT / profile["level"]["config"]
    family = profile["model_family"]
    env = make_env(config, family)
    env.reset(seed=0)
    env.close()
    suite, suite_path = load_suite(DEFAULT_SUITE_PATH)
    if profile["level"]["id"] not in {level["id"] for level in suite["levels"]}:
        raise ValueError(f"El nivel {profile['level']['id']} no está en la suite {suite['evaluation_suite_id']}")
    criteria = yaml.safe_load(CRITERIA_PATH.read_text(encoding="utf-8"))
    if profile["level"]["id"] not in criteria["min_success_rate"]:
        raise ValueError(f"No hay criterio de selección para {profile['level']['id']}")
    _write_state(state_path, "validate", {
        "profile": str(resolved.relative_to(PROJECT_ROOT)) if resolved.is_relative_to(PROJECT_ROOT) else str(resolved),
        "profile_sha256": _sha256(resolved), "level": profile["level"]["id"], "family": family,
        "suite": suite["evaluation_suite_id"], "suite_sha256": _sha256(suite_path),
        "criteria_version": criteria["version"], "environment_version": ENVIRONMENT_VERSION,
    }, level=profile["level"]["id"], family=family)


def tests() -> None:
    """Etapa 2: reglas, observaciones, acciones y terminaciones."""
    subprocess.run([sys.executable, "-m", "pytest", "-q", *FAST_TESTS], cwd=PROJECT_ROOT, check=True)


def train(state_path: str, profile_path: str, seed: int, timesteps: int | None, run_id: str | None) -> None:
    """Etapa 3: entrena un candidato, o toma un run ya existente y compatible."""
    if run_id:
        manifest = _manifest(run_id)
        if manifest.get("model_sha256") is None:
            raise ValueError(f"El run {run_id} no tiene checkpoint: {manifest.get('failure_detail')}")
        _write_state(state_path, "train", {"origin": "run existente", "run_id": run_id}, run_id=run_id)
        return
    result = train_run(profile_path, seed, timesteps_override=timesteps)
    _write_state(state_path, "train", {"origin": "entrenado", "run_id": result["run_id"], "registry_uri": result["registry_uri"]},
                 run_id=result["run_id"])


def evaluate(state_path: str) -> None:
    """Etapa 4: repite la suite con el modelo congelado y comprueba que coincide con lo registrado."""
    state = _read_state(state_path)
    run_id = state["run_id"]
    manifest = _manifest(run_id)
    package = PROJECT_ROOT / manifest["package_path"]
    if _sha256(package / "model.zip") != manifest["model_sha256"]:
        raise ValueError("El modelo del paquete no coincide con el manifiesto")
    output = RESULTS_DIR / f"evaluacion-{run_id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    result = evaluate_model(DEFAULT_SUITE_PATH, package / "model.zip", manifest["model_family"], output)
    registered = json.loads((package / "evaluacion.json").read_text(encoding="utf-8"))
    summary = {case["case_id"]: case["metrics"]["success_rate"] for case in result["cases"]}
    recorded = {case["case_id"]: case["metrics"]["success_rate"] for case in registered["cases"]}
    if summary != recorded:
        raise ValueError(f"La evaluación no reproduce la registrada: {summary} frente a {recorded}")
    _write_state(state_path, "evaluate", {"output": str(output.relative_to(PROJECT_ROOT)), "success_by_case": summary, "reproduces_recorded": True},
                 evaluation_output=str(output))


def register(state_path: str) -> None:
    """Etapa 5: comprueba hashes y procedencia del modelo en MLflow."""
    state = _read_state(state_path)
    manifest = _manifest(state["run_id"])
    downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=manifest["model_uri"]))
    if _sha256(downloaded) != manifest["model_sha256"]:
        raise ValueError("El modelo en MLflow no coincide con el manifiesto")
    version = MlflowClient().get_model_version(manifest["registry_name"], manifest["registry_version"])
    _write_state(state_path, "register", {
        "registry_uri": manifest["registry_uri"], "registry_status": version.status,
        "model_sha256": manifest["model_sha256"], "git_commit": manifest["git_commit"], "git_dirty": manifest["git_dirty"],
    })


def decide(evaluation: dict, level: str, criteria: dict, current: dict | None) -> dict:
    """Aplica los criterios fijados. Devuelve la decisión y todos sus motivos."""
    case = next(case for case in evaluation["cases"] if case["level_id"] == level)["metrics"]
    success, steps = float(case["success_rate"]), float(case["mean_steps"])
    reasons = []
    minimum = float(criteria["min_success_rate"][level])
    if success < minimum:
        reasons.append(f"éxito {success:.0%} por debajo del mínimo {minimum:.0%}")
    if steps > float(criteria["max_mean_steps"]):
        reasons.append(f"{steps:.0f} pasos medios superan el máximo {criteria['max_mean_steps']}")
    if criteria.get("no_regression") and current:
        if (success, -steps) < (float(current["success_rate"]), -float(current["mean_steps"])):
            reasons.append(f"no mejora al vigente {current['run_id']} ({float(current['success_rate']):.0%}, {float(current['mean_steps']):.0f} pasos)")
    return {"decision": "aceptado" if not reasons else "rechazado", "reasons": reasons,
            "success_rate": success, "mean_steps": steps, "cause_counts": case["cause_counts"]}


def published_versions(level: str, directory: Path = PUBLISHED_DIR) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((directory / level).glob("v*.json"))]


def current_published(level: str, directory: Path = PUBLISHED_DIR) -> dict | None:
    versions = published_versions(level, directory)
    return versions[-1] if versions else None


def write_publication(level: str, entry: dict, directory: Path = PUBLISHED_DIR) -> dict:
    """Nueva versión explícita; nunca sobrescribe. Repetir la vigente no crea otra."""
    current = current_published(level, directory)
    if current and current["model_sha256"] == entry["model_sha256"]:
        return {**current, "already_published": True}
    number = len(published_versions(level, directory)) + 1
    record = {"publication_version": number, "level": level, "published_at": _now(), **entry}
    path = directory / level / f"v{number:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def _set_alias(entry: dict) -> None:
    MlflowClient().set_registered_model_alias(entry["registry_name"], PRODUCTION_ALIAS, entry["registry_version"])


def select(state_path: str) -> None:
    """Etapa 6: criterios fijados previamente; rechazar deja el candidato y el motivo."""
    state = _read_state(state_path)
    manifest = _manifest(state["run_id"])
    criteria = yaml.safe_load(CRITERIA_PATH.read_text(encoding="utf-8"))
    evaluation = json.loads(Path(state["evaluation_output"]).read_text(encoding="utf-8"))
    decision = decide(evaluation, manifest["level_id"], criteria, current_published(manifest["level_id"]))
    decision["criteria_sha256"] = _sha256(CRITERIA_PATH)
    mlflow.MlflowClient().set_tag(state["run_id"], "selection_decision", decision["decision"])
    _write_state(state_path, "select", decision, decision=decision["decision"])


def publish(state_path: str) -> None:
    """Etapa 7: publica la referencia al modelo recomendado solo si fue aceptado."""
    state = _read_state(state_path)
    if state["decision"] != "aceptado":
        _write_state(state_path, "publish", {"published": False, "motivo": state["stages"]["select"]["reasons"]})
        return
    manifest = _manifest(state["run_id"])
    select_stage = state["stages"]["select"]
    entry = write_publication(manifest["level_id"], {
        "action": "promocion",
        "run_id": state["run_id"],
        "model_family": manifest["model_family"],
        "model_sha256": manifest["model_sha256"],
        "registry_name": manifest["registry_name"],
        "registry_version": manifest["registry_version"],
        "registry_uri": manifest["registry_uri"],
        "git_commit": manifest["git_commit"],
        "success_rate": select_stage["success_rate"],
        "mean_steps": select_stage["mean_steps"],
    })
    _set_alias(entry)
    _write_state(state_path, "publish", {"published": True, "publication_version": entry["publication_version"], "run_id": entry["run_id"]})


def verify(level: str, state_path: str | None = None) -> dict:
    """Etapa 8: carga por la API la versión publicada y comprueba cuál se sirve realmente."""
    from fastapi.testclient import TestClient

    from api.main import _create_app, runtime

    entry = current_published(level)
    if entry is None:
        evidence = {"level": level, "verified": False, "motivo": "no hay modelo publicado"}
        if state_path:
            _write_state(state_path, "verify", evidence)
        print(json.dumps(evidence, ensure_ascii=False))
        return evidence
    client = TestClient(_create_app(production=False, cors_origins=[]))
    loaded = client.post("/load", json={"run_id": entry["run_id"], "seed": 0})
    if loaded.status_code != 200:
        raise RuntimeError(f"La API no pudo cargar {entry['run_id']}: {loaded.text}")
    headers = {"X-Drone-Session": loaded.json()["session_id"]}
    served = client.get("/version", headers=headers).json()["served_model"]
    if not served or served["model_sha256"] != entry["model_sha256"]:
        raise RuntimeError("El modelo servido no es el publicado")
    client.post("/episode/reset", json={"mode": "evaluacion", "seed": 101}, headers=headers)
    state = {"terminated": False, "truncated": False}
    while not (state["terminated"] or state["truncated"]):
        state = client.get("/episode/step", headers=headers).json()
    runtime.close_session(headers["X-Drone-Session"])
    evidence = {"level": level, "verified": True, "publication_version": entry["publication_version"], "run_id": entry["run_id"],
                "served_sha256": served["model_sha256"], "smoke_episode_cause": state["cause"], "smoke_steps": state["telemetry"]["steps"]}
    if state_path:
        _write_state(state_path, "verify", evidence)
    print(json.dumps(evidence, ensure_ascii=False))
    return evidence


def rollback(level: str, directory: Path = PUBLISHED_DIR, apply_alias: bool = True) -> dict:
    """Restaura la versión anterior como una versión nueva, conservando todo el historial."""
    versions = published_versions(level, directory)
    if len(versions) < 2:
        raise ValueError(f"{level} no tiene una versión anterior a la que volver")
    current, previous = versions[-1], versions[-2]
    fields = ("run_id", "model_family", "model_sha256", "registry_name", "registry_version", "registry_uri", "git_commit", "success_rate", "mean_steps")
    entry = write_publication(level, {"action": "rollback", "restored_from_version": previous["publication_version"],
                                      "replaced_version": current["publication_version"],
                                      **{field: previous[field] for field in fields}}, directory)
    if apply_alias:
        _set_alias(entry)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["validate", "tests", "train", "evaluate", "register", "select", "publish", "verify", "rollback"])
    parser.add_argument("--state")
    parser.add_argument("--profile")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--level")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    configure_tracking()
    if args.stage == "validate":
        validate(args.state, args.profile)
    elif args.stage == "tests":
        tests()
    elif args.stage == "train":
        train(args.state, args.profile, args.seed, args.timesteps or None, args.run_id or None)
    elif args.stage == "evaluate":
        evaluate(args.state)
    elif args.stage == "register":
        register(args.state)
    elif args.stage == "select":
        select(args.state)
    elif args.stage == "publish":
        publish(args.state)
    elif args.stage == "verify":
        verify(args.level or _read_state(args.state).get("level"), args.state)
    elif args.stage == "rollback":
        entry = rollback(args.level)
        print(json.dumps(entry, ensure_ascii=False))
        verify(args.level)


if __name__ == "__main__":
    main()
