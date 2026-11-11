"""Entrenamiento versionado de un DQN de la familia de cuadrícula (grid-drone-v2).

Mismo contrato que pipelines/train_ppo.py: rechaza código sin commit, registra el run en MLflow
(parámetros, curvas, artefactos y Model Registry), evalúa el mejor modelo con la suite fija y
deja el paquete recuperable en artifacts/models/<run_id>/ y el manifiesto en manifests/.

    python -m pipelines.train_grid --profile configs/training/grid_storm_dqn.yaml --seed 11
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from mlflow.models import infer_signature

import envs.grid_drone_env as grid
from agents.grid.dqn import DQNAgent, train_dqn
from agents.grid.dqn_model import GridDQNPyFuncModel
from agents.grid.evaluate import evaluate_storm, fmt_storm, score_storm
from pipelines.tracking import configure_tracking
from pipelines.train import _json_bytes, _write_manifest
from pipelines.train_ppo import DEFAULT_MANIFEST_DIR, DEFAULT_PACKAGE_DIR, PROJECT_ROOT, _git, _sha256

ACTIONS = ["+x", "-x", "+y", "-y", "subir", "bajar"]


def input_contract() -> dict:
    """Contrato de entrada y salida (sección 11 de la guía), en el orden de DroneStormEnv._obs."""
    return {
        "family": grid.STORM_FAMILY,
        "environment_version": grid.STORM_ENVIRONMENT_VERSION,
        "observation_dimensions": grid.STORM_N_OBS,
        "action_dimensions": len(ACTIONS),
        "actions": ACTIONS,
        "action_unit": "una celda de 5 m",
        "features": [
            {"indices": "0:3", "from": "posición (celdas)", "scale": "/ (19, 19, 11)"},
            {"indices": "3:6", "from": "vector hacia B (celdas)", "scale": "/ (19, 19, 11)"},
            {"indices": "6:12", "from": "sensores de distancia +x, -x, +y, -y, arriba, abajo", "scale": f"/ {grid.SENSOR_RANGE} celdas"},
            {"indices": "12:39", "from": "ocupación del cubo 3x3x3 alrededor del dron", "scale": "1 = bloqueado"},
            {"indices": "39:46", "from": "visitas a la celda actual y a las 6 vecinas", "scale": "/ 3, máximo 1"},
            {"indices": "46:48", "from": "viento: dirección x fuerza", "scale": f"/ {grid.WIND_MAX}"},
            {"indices": "48", "from": "probabilidad de ráfaga a la altura actual", "scale": f"/ {grid.WIND_MAX}"},
            {"indices": "49", "from": "batería restante", "scale": "/ capacidad"},
            {"indices": "50", "from": "batería menos energía estimada hasta B", "scale": "/ 30, clip [-1, 1]"},
        ],
        "contract": "docs/familia_cuadricula.md",
    }


def load_profile(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("model_family", "experiment", "environment", "training", "evaluation_suite"):
        if not profile.get(key):
            raise ValueError(f"Falta {key} en el perfil")
    if profile["model_family"] != grid.STORM_FAMILY:
        raise ValueError(f"Este pipeline entrena {grid.STORM_FAMILY}, no {profile['model_family']}")
    return profile, path


def load_grid_suite(path: str | Path) -> tuple[dict, Path]:
    path = (PROJECT_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("evaluation_suite_id", "version", "environment", "first_seed", "episodes"):
        if key not in suite:
            raise ValueError(f"Falta {key} en la suite {path}")
    return suite, path


def train_run(profile_path: str | Path, seed: int, *, manifest_dir: str | Path = DEFAULT_MANIFEST_DIR,
              package_dir: str | Path = DEFAULT_PACKAGE_DIR, steps_override: int | None = None,
              allow_dirty: bool = False) -> dict:
    profile, profile_file = load_profile(profile_path)
    git_commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--", ".", ":!manifests", ":!artifacts"))
    if dirty and not allow_dirty:
        raise RuntimeError("Hay cambios sin commit: el run no identificaría el código. Haz commit o usa --allow-dirty")
    training = dict(profile["training"])
    if steps_override is not None:
        training["steps"] = int(steps_override)
    agent_params = dict(training.get("agent", {}))
    env_id = str(profile["environment"]["id"])
    experiment_id = str(profile["experiment"]["id"])
    suite, suite_path = load_grid_suite(profile["evaluation_suite"])
    if suite["environment"] != env_id:
        raise ValueError(f"La suite {suite['evaluation_suite_id']} es de {suite['environment']}, no de {env_id}")
    validation = training["validation"]
    requirements_sha256 = _sha256(PROJECT_ROOT / "requirements.txt")
    mlflow.set_experiment(str(profile.get("mlflow_experiment", "drone-rescue-training")))
    tags = {
        "git_commit": git_commit, "git_dirty": str(dirty).lower(), "environment_version": grid.STORM_ENVIRONMENT_VERSION,
        "model_family": grid.STORM_FAMILY, "algorithm": "dqn", "level_id": env_id, "experiment_id": experiment_id,
        "hypothesis": str(profile["experiment"]["hypothesis"]), "parent_run_id": "none",
        "evaluation_suite_id": str(suite["evaluation_suite_id"]), "checkpoint_status": "sin checkpoint disponible",
    }
    manifest = {
        "run_id": None, "level_id": env_id, "experiment_id": experiment_id, "algorithm": "dqn",
        "model_family": grid.STORM_FAMILY, "environment_version": grid.STORM_ENVIRONMENT_VERSION,
        "reward_version": grid.STORM_REWARD_VERSION, "git_commit": git_commit, "git_dirty": dirty,
        "config_sha256": None, "config_uri": None, "model_uri": None, "model_sha256": None,
        "normalization_uri": None, "normalization_sha256": None,
        "runtime_ref": (f"python=={platform.python_version()}; numpy=={np.__version__}; torch=={torch.__version__}; "
                        f"gymnasium=={version('gymnasium')}; scipy=={version('scipy')}; mlflow=={mlflow.__version__}"),
        "requirements_sha256": requirements_sha256,
        "evaluation_suite_id": str(suite["evaluation_suite_id"]), "evaluation_suite_version": str(suite["version"]),
        "evaluation_suite_sha256": _sha256(suite_path), "seed": int(seed), "parent_run_id": None,
        "effective_timesteps": None,
        "resumable_checkpoint": "final.pt incluye red, red objetivo, optimizador y contador de pasos",
        "serving": "se usa con cli.py (familia de cuadrícula); la API sirve drone-rescue-v2 y v3",
        "diagnostic": "sin checkpoint disponible",
    }
    with mlflow.start_run(run_name=f"{experiment_id}-seed-{int(seed)}", tags=tags) as run:
        run_id = run.info.run_id
        manifest["run_id"] = run_id
        started = time.time()
        try:
            snapshot = {
                "profile": profile,
                "profile_path": str(profile_file.relative_to(PROJECT_ROOT)) if profile_file.is_relative_to(PROJECT_ROOT) else str(profile_file),
                "environment": {"id": env_id, "family": grid.STORM_FAMILY, "version": grid.STORM_ENVIRONMENT_VERSION,
                                "reward_version": grid.STORM_REWARD_VERSION, "cell_m": grid.CELL, "size": [grid.NX, grid.NY, grid.NZ],
                                "start": list(grid.START), "columns": list(grid.STORM_COLUMNS), "goal_min_dist": grid.GOAL_MIN_DIST,
                                "wind_max": grid.WIND_MAX, "energy": {"horizontal": grid.ENERGY_H, "subir": grid.ENERGY_UP,
                                                                      "bajar": grid.ENERGY_DOWN, "headwind": grid.HEADWIND,
                                                                      "gust": grid.GUST_ENERGY},
                                "battery_margin": list(grid.BATTERY_MARGIN), "battery_reserve": grid.BATTERY_RESERVE,
                                "rewards": {"goal": grid.R_GOAL, "crash": grid.R_CRASH, "battery": grid.R_BATTERY,
                                            "progress": grid.R_PROGRESS, "low": grid.R_LOW}},
                "seed": int(seed), "steps": int(training["steps"]), "evaluation_suite_id": suite["evaluation_suite_id"],
            }
            config_bytes = _json_bytes(snapshot)
            config_sha256 = hashlib.sha256(config_bytes).hexdigest()
            manifest["config_sha256"] = config_sha256
            mlflow.log_params({"experiment_id": experiment_id, "level_id": env_id, "seed": int(seed),
                               "model_family": grid.STORM_FAMILY, "algorithm": "dqn", "timesteps": int(training["steps"]),
                               "environment_version": grid.STORM_ENVIRONMENT_VERSION, "reward_version": grid.STORM_REWARD_VERSION,
                               "validation_first_seed": int(validation["first_seed"]), "validation_episodes": int(validation["episodes"]),
                               "config_sha256": config_sha256, "requirements_sha256": requirements_sha256,
                               **{f"dqn_{key}": value for key, value in agent_params.items()}})
            torch.set_num_threads(int(training.get("torch_threads", 2)))
            with tempfile.TemporaryDirectory(prefix=f"drone-grid-{run_id}-") as tmp:
                temp = Path(tmp)
                curve = []

                eval_every = int(training["eval_every"])

                def eval_fn(agent):
                    result = evaluate_storm(agent, env_id, episodes=int(validation["episodes"]), seed=int(validation["first_seed"]))
                    metrics = result["metrics"]
                    step = (len(curve) + 1) * eval_every
                    curve.append({"step": step, **{key: metrics[key] for key in (
                        "success_rate", "crash_rate", "wind_crash_rate", "battery_rate", "timeout_rate", "energy_ratio")}})
                    mlflow.log_metrics({f"validation/{key}": float(value) for key, value in curve[-1].items()
                                        if key != "step" and value is not None}, step=step)
                    return "validación " + fmt_storm(metrics), score_storm(metrics)

                best_path = temp / "model.pt"
                agent = train_dqn(env_id, steps=int(training["steps"]), device=str(training.get("device", "cpu")), seed=int(seed),
                                  log_every=eval_every, eval_fn=eval_fn, save_path=best_path,
                                  on_log=lambda step, values: mlflow.log_metrics(values, step=step), **agent_params)
                if not best_path.exists():  # presupuesto menor que eval_every: no hubo validación
                    agent.save(best_path)
                final_path = temp / "final.pt"
                agent.save_full(final_path, int(training["steps"]))
                config_artifact = temp / "config.json"
                config_artifact.write_bytes(config_bytes)
                normalization_path = temp / "normalization.json"
                normalization_path.write_bytes(_json_bytes(input_contract()))

                best = DQNAgent.load(best_path, "cpu")
                evaluation = evaluate_storm(best, env_id, episodes=int(suite["episodes"]), seed=int(suite["first_seed"]))
                evaluation = {"evaluation_suite_id": suite["evaluation_suite_id"], "evaluation_suite_version": suite["version"],
                              "evaluation_suite_sha256": _sha256(suite_path), "environment": env_id, "split": suite.get("split"),
                              "model_family": grid.STORM_FAMILY, "cases": [{"case_id": f"{env_id}-dqn", "level_id": env_id,
                                                                            "policy_id": "dqn", **evaluation}],
                              "validation_curve": curve}
                evaluation_path = temp / "evaluacion.json"
                evaluation_path.write_bytes(_json_bytes(evaluation))
                metrics = evaluation["cases"][0]["metrics"]
                mlflow.log_metrics({f"final_evaluation/{key}": float(value) for key, value in metrics.items()
                                    if isinstance(value, (int, float)) and value is not None}, step=int(training["steps"]))

                example, _ = grid.DroneStormEnv().reset(seed=int(seed))
                input_example = np.asarray([example], dtype=np.float32)
                model_info = mlflow.pyfunc.log_model(
                    name="pyfunc_model", python_model=GridDQNPyFuncModel(),
                    artifacts={"model": str(best_path), "normalization": str(normalization_path)},
                    code_paths=[str(PROJECT_ROOT / "agents"), str(PROJECT_ROOT / "envs")],
                    pip_requirements=[f"mlflow=={mlflow.__version__}", f"numpy=={np.__version__}", f"torch=={torch.__version__}",
                                      f"gymnasium=={version('gymnasium')}", f"scipy=={version('scipy')}"],
                    input_example=input_example, signature=infer_signature(input_example, np.zeros((1,), dtype=np.int64)),
                )
                registered_name = f"DroneRescueGridDQN-{grid.STORM_FAMILY}"
                registered = mlflow.register_model(model_info.model_uri, registered_name)
                registry_uri = f"models:/{registered_name}/{registered.version}"

                package = Path(package_dir) / run_id
                if package.exists():
                    raise FileExistsError(f"El paquete del modelo ya existe: {package}")
                package.mkdir(parents=True)
                for source in (best_path, final_path, config_artifact, normalization_path, evaluation_path):
                    shutil.copyfile(source, package / source.name)
                for name, folder in (("model.pt", "checkpoints"), ("final.pt", "checkpoints"), ("config.json", "config"),
                                     ("normalization.json", "normalization"), ("evaluacion.json", "evaluation")):
                    mlflow.log_artifact(str(package / name), artifact_path=folder)
                manifest.update({
                    "config_uri": f"runs:/{run_id}/config/config.json",
                    "model_uri": f"runs:/{run_id}/checkpoints/model.pt", "model_sha256": _sha256(package / "model.pt"),
                    "final_checkpoint_uri": f"runs:/{run_id}/checkpoints/final.pt",
                    "final_checkpoint_sha256": _sha256(package / "final.pt"),
                    "normalization_uri": f"runs:/{run_id}/normalization/normalization.json",
                    "normalization_sha256": _sha256(package / "normalization.json"),
                    "evaluation_uri": f"runs:/{run_id}/evaluation/evaluacion.json",
                    "evaluation_sha256": _sha256(package / "evaluacion.json"),
                    "package_path": str(package.relative_to(PROJECT_ROOT)) if package.is_relative_to(PROJECT_ROOT) else str(package),
                    "registry_uri": registry_uri, "registry_name": registered_name, "registry_version": str(registered.version),
                    "effective_timesteps": int(training["steps"]), "training_duration_s": round(time.time() - started, 1),
                    "diagnostic": None,
                })
                manifest_path = _write_manifest(Path(manifest_dir), run_id, manifest)
                mlflow.log_artifact(str(manifest_path), artifact_path="manifest")
                mlflow.set_tags({"checkpoint_status": "disponible", "registry_uri": registry_uri, "config_sha256": config_sha256,
                                 "checkpoint_artifact_path": "checkpoints/final.pt"})
                return {"run_id": run_id, "manifest_path": str(manifest_path), "registry_uri": registry_uri, "metrics": metrics}
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            mlflow.set_tags({"checkpoint_status": "sin checkpoint disponible", "diagnostic": "sin checkpoint disponible",
                             "failure_detail": detail[:500]})
            mlflow.log_text(f"sin checkpoint disponible\n{detail}\n", "diagnostic.txt")
            manifest["failure_detail"] = detail
            try:
                _write_manifest(Path(manifest_dir), run_id, manifest)
            except FileExistsError:
                pass
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, help="Sobrescribe el presupuesto del perfil; queda en el manifiesto")
    parser.add_argument("--allow-dirty", action="store_true", help="Solo pruebas: marca el run con git_dirty=true")
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE_DIR))
    args = parser.parse_args()
    configure_tracking()
    result = train_run(args.profile, args.seed, manifest_dir=args.manifest_dir, package_dir=args.package_dir,
                       steps_override=args.steps, allow_dirty=args.allow_dirty)
    print(json.dumps({key: result[key] for key in ("run_id", "manifest_path", "registry_uri", "metrics")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
