"""Entrenamiento PPO reproducible, con MLflow, manifiesto y paquete recuperable del modelo.

    python -m pipelines.train_ppo --profile configs/training/ppo_v3_n1.yaml --seed 11
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
import time
from collections import Counter, deque
from functools import partial
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from mlflow.models import infer_signature
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from agents.ppo import PPOPolicy, input_contract, make_env
from agents.ppo_model import PPOPyFuncModel
from envs.drone_env import ENVIRONMENT_VERSION, load_config
from envs.stabilized_drone_env import STABILIZER_VERSION
from pipelines.evaluate import DEFAULT_SUITE_PATH, _resolve_project_path, _summarize_case, load_suite
from pipelines.tracking import configure_tracking
from pipelines.train import _json_bytes, _project_path, _sha256, _write_manifest, load_profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "manifests"
DEFAULT_PACKAGE_DIR = PROJECT_ROOT / "artifacts" / "models"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def run_policy(env, policy, seed: int) -> dict:
    """Un episodio del modelo congelado; mismos campos que pipelines.evaluate."""
    features, _ = env.reset(seed=seed)
    drone = env.unwrapped
    total, latencies, info, done = 0.0, [], {"cause": None}, False
    precision = speed = None
    while not done:
        started = time.perf_counter()
        action = policy.act(features)
        latencies.append(time.perf_counter() - started)
        stage_before = drone.stage
        at_release = drone.position.copy(), float(np.linalg.norm(drone.velocity))
        features, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        done = terminated or truncated
        if stage_before == "delivery" and drone.stage == "return":
            precision = float(np.linalg.norm((at_release[0] - drone._delivery)[:2]))
            speed = at_release[1]
    return {
        "seed": seed,
        "success": info["cause"] == "success",
        "cause": info["cause"],
        "stage": drone.stage,
        "steps": int(drone.steps),
        "battery_remaining_wh": float(drone.battery_wh),
        "delivery_precision_m": precision,
        "delivery_speed_mps": speed,
        "collisions": int(drone.contacts),
        "return": total,
        "inference_latency_ms": float(np.mean(latencies) * 1000.0) if latencies else 0.0,
        "total_inference_latency_ms": float(np.sum(latencies) * 1000.0),
    }


def evaluate_model(suite_path: str | Path, model_path: str | Path, family: str, output_path: str | Path) -> dict:
    """Evalúa el modelo congelado en la suite, siempre sobre ``DroneEnv``, con resultados por caso."""
    suite, resolved = load_suite(suite_path)
    policy = PPOPolicy(model_path)
    cases = []
    for level in suite["levels"]:
        config_path = _resolve_project_path(level["config"])
        reward_version = str(load_config(config_path)["reward"]["version"])
        env = make_env(config_path, family)
        episodes = [{"level_id": level["id"], "policy_id": "ppo", "reward_version": reward_version,
                     **run_policy(env, policy, int(seed))} for seed in suite["seeds"]]
        env.close()
        cases.append({
            "case_id": f"{level['id']}-ppo",
            "level_id": level["id"],
            "config_path": str(level["config"]),
            "config_sha256": _sha256(config_path),
            "reward_version": reward_version,
            "policy_id": "ppo",
            "episodes": episodes,
            "metrics": _summarize_case(episodes, reward_version),
        })
    result = {
        "evaluation_suite_id": suite["evaluation_suite_id"],
        "evaluation_suite_version": suite["version"],
        "split": suite["split"],
        "evaluation_suite_sha256": _sha256(resolved),
        "environment_version": ENVIRONMENT_VERSION,
        "model_family": family,
        "seeds": [int(seed) for seed in suite["seeds"]],
        "model_sha256": _sha256(Path(model_path)),
        "return_comparison_scope": "solo dentro del mismo nivel y reward_version; sin agregado entre casos",
        "cases": cases,
    }
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"La salida ya existe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_bytes(result))
    return result


def mission_score(episodes: list[dict]) -> tuple:
    """Selección por misión: éxito, luego entrega, luego menos pasos. El retorno no interviene."""
    return (
        float(np.mean([e["success"] for e in episodes])),
        float(np.mean([e["stage"] in ("return", "complete") for e in episodes])),
        -float(np.mean([e["steps"] for e in episodes])),
    )


class ProgressCallback(BaseCallback):
    """Curvas separadas de entrenamiento (con exploración) y evaluación (sin ella)."""

    def __init__(self, config_path: Path, family: str, seeds: list[int], every: int, best_path: Path, status_path: Path | None):
        super().__init__()
        self.config_path, self.family, self.seeds, self.every = config_path, family, seeds, every
        self.best_path, self.status_path = best_path, status_path
        self.next_eval, self.best, self.best_metrics = every, None, None
        self.recent = deque(maxlen=200)
        self.evaluation_curve, self.training_curve = [], []
        self.started = time.time()

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.recent.append((info["cause"], info["stage"]))
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        step = int(self.num_timesteps)
        n = max(1, len(self.recent))
        causes = Counter(cause for cause, _ in self.recent)
        training = {
            "training/success": causes["success"] / n,
            "training/delivered": sum(stage in ("return", "complete") for _, stage in self.recent) / n,
            "training/collision": causes["collision"] / n,
            "training/out_of_bounds": causes["out_of_bounds"] / n,
            "training/step_limit": causes["step_limit"] / n,
        }
        self.model.save(self.best_path.with_name("current.zip"))
        policy = PPOPolicy(self.best_path.with_name("current.zip"))
        env = make_env(self.config_path, self.family, fast=True)
        episodes = [run_policy(env, policy, seed) for seed in self.seeds]
        env.close()
        evaluation = {
            "evaluation/success": float(np.mean([e["success"] for e in episodes])),
            "evaluation/delivered": float(np.mean([e["stage"] in ("return", "complete") for e in episodes])),
            "evaluation/steps": float(np.mean([e["steps"] for e in episodes])),
            "evaluation/return": float(np.mean([e["return"] for e in episodes])),
        }
        mlflow.log_metrics({**training, **evaluation}, step=step)
        self.training_curve.append({"timesteps": step, **training})
        self.evaluation_curve.append({"timesteps": step, **evaluation})
        score = mission_score(episodes)
        if self.best is None or score > self.best:
            self.best, self.best_metrics = score, evaluation
            shutil.copyfile(self.best_path.with_name("current.zip"), self.best_path)
        line = {"timesteps": step, "elapsed_s": round(time.time() - self.started), **training, **evaluation}
        print(json.dumps(line), flush=True)
        if self.status_path:
            self.status_path.write_text(json.dumps(line), encoding="utf-8")


def train_run(
    profile_path: str | Path,
    seed: int,
    *,
    parent_run_id: str | None = None,
    manifest_dir: str | Path = DEFAULT_MANIFEST_DIR,
    package_dir: str | Path = DEFAULT_PACKAGE_DIR,
    suite_path: str | Path | None = None,
    timesteps_override: int | None = None,
    allow_dirty: bool = False,
    status_path: str | Path | None = None,
) -> dict:
    profile, profile_file = load_ppo_profile(profile_path)
    git_commit = _git("rev-parse", "HEAD")
    # Las salidas de runs anteriores no cambian el código que se ejecuta
    dirty = bool(_git("status", "--porcelain", "--", ".", ":!manifests", ":!artifacts"))
    if dirty and not allow_dirty:
        raise RuntimeError("Hay cambios sin commit: el run no identificaría el código. Haz commit o usa --allow-dirty")
    training = dict(profile["training"])
    if timesteps_override is not None:
        training["timesteps"] = int(timesteps_override)
    experiment_id, level_id = str(profile["experiment"]["id"]), str(profile["level"]["id"])
    family = str(profile["model_family"])
    hypothesis = str(profile["experiment"]["hypothesis"])
    config_path = _project_path(profile["level"]["config"])
    suite_file = Path(suite_path).resolve() if suite_path else DEFAULT_SUITE_PATH
    suite, _ = load_suite(suite_file)
    requirements_sha256 = _sha256(PROJECT_ROOT / "requirements.txt")
    mlflow.set_experiment(str(profile.get("mlflow_experiment", "drone-rescue-training")))
    tags = {
        "git_commit": git_commit,
        "git_dirty": str(dirty).lower(),
        "environment_version": ENVIRONMENT_VERSION,
        "model_family": family,
        "algorithm": "ppo",
        "level_id": level_id,
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "parent_run_id": str(parent_run_id or "none"),
        "evaluation_suite_id": str(suite["evaluation_suite_id"]),
        "checkpoint_status": "sin checkpoint disponible",
    }
    manifest = {
        "run_id": None,
        "level_id": level_id,
        "experiment_id": experiment_id,
        "algorithm": "ppo",
        "model_family": family,
        "environment_version": ENVIRONMENT_VERSION,
        "stabilizer_version": STABILIZER_VERSION if family == "drone-rescue-v3" else None,
        "git_commit": git_commit,
        "git_dirty": dirty,
        "config_sha256": None,
        "config_uri": None,
        "model_uri": None,
        "model_sha256": None,
        "normalization_uri": None,
        "normalization_sha256": None,
        "runtime_ref": (
            f"python=={platform.python_version()}; numpy=={np.__version__}; torch=={torch.__version__}; "
            f"gymnasium=={version('gymnasium')}; stable-baselines3=={version('stable-baselines3')}; "
            f"mlflow=={mlflow.__version__}"
        ),
        "requirements_sha256": requirements_sha256,
        "evaluation_suite_id": str(suite["evaluation_suite_id"]),
        "evaluation_suite_version": str(suite["version"]),
        "evaluation_suite_sha256": _sha256(suite_file),
        "seed": int(seed),
        "parent_run_id": parent_run_id,
        "effective_timesteps": None,
        "resumable_checkpoint": "el .zip incluye optimizador y contador de pasos",
        "diagnostic": "sin checkpoint disponible",
    }
    with mlflow.start_run(run_name=f"{experiment_id}-seed-{int(seed)}", tags=tags) as run:
        run_id = run.info.run_id
        manifest["run_id"] = run_id
        started = time.time()
        vector_env = None
        try:
            environment_config = load_config(config_path)
            snapshot = {
                "profile": profile,
                "profile_path": str(profile_file.relative_to(PROJECT_ROOT)) if profile_file.is_relative_to(PROJECT_ROOT) else str(profile_file),
                "environment": environment_config,
                "seed": int(seed),
                "parent_run_id": parent_run_id,
                "evaluation_suite_id": suite["evaluation_suite_id"],
            }
            config_bytes = _json_bytes(snapshot)
            config_sha256 = hashlib.sha256(config_bytes).hexdigest()
            manifest["config_sha256"] = config_sha256
            agent = training["agent"]
            mlflow.log_params({
                "experiment_id": experiment_id, "level_id": level_id, "seed": int(seed),
                "model_family": family, "algorithm": "ppo",
                "level_config": str(profile["level"]["config"]),
                "environment_version": ENVIRONMENT_VERSION,
                "reward_version": str(environment_config["reward"]["version"]),
                "timesteps": int(training["timesteps"]),
                "n_envs": int(training["n_envs"]),
                "evaluation_seeds": json.dumps(training["evaluation_seeds"]),
                "config_sha256": config_sha256,
                "requirements_sha256": requirements_sha256,
                **{f"ppo_{key}": value for key, value in agent.items()},
            })
            with tempfile.TemporaryDirectory(prefix=f"drone-rescue-{run_id}-") as tmp:
                temp = Path(tmp)
                vector_env = SubprocVecEnv([
                    partial(_make_worker, str(config_path), family, int(seed) + index)
                    for index in range(int(training["n_envs"]))
                ])
                if parent_run_id:
                    parent_model = _parent_checkpoint(parent_run_id, level_id, family)
                    model = PPO.load(str(parent_model), env=vector_env, device="cpu", seed=int(seed))
                else:
                    kwargs = dict(agent)
                    net_arch = kwargs.pop("net_arch")
                    model = PPO("MlpPolicy", vector_env, device="cpu", seed=int(seed), verbose=0,
                                policy_kwargs={"net_arch": net_arch}, **kwargs)
                callback = ProgressCallback(
                    config_path, family, [int(s) for s in training["evaluation_seeds"]],
                    int(training["evaluation_every"]), temp / "best.zip",
                    Path(status_path) if status_path else None,
                )
                model.learn(int(training["timesteps"]), callback=callback, reset_num_timesteps=not parent_run_id)
                final_path = temp / "final.zip"
                model.save(final_path)
                effective = int(model.num_timesteps)
                vector_env.close()
                vector_env = None
                best_path = temp / "best.zip" if (temp / "best.zip").exists() else final_path
                model_path = temp / "model.zip"
                shutil.copyfile(best_path, model_path)

                config_artifact = temp / "config.json"
                config_artifact.write_bytes(config_bytes)
                normalization_path = temp / "normalization.json"
                normalization_path.write_bytes(_json_bytes(input_contract(family)))
                evaluation_path = temp / "evaluacion.json"
                evaluation = evaluate_model(suite_file, model_path, family, evaluation_path)
                evaluation["training_curve"] = callback.training_curve
                evaluation["evaluation_curve"] = callback.evaluation_curve
                evaluation_path.write_bytes(_json_bytes(evaluation))
                for case in evaluation["cases"]:
                    metrics = case["metrics"]
                    prefix = f"final_evaluation/{case['case_id']}"
                    mlflow.log_metrics({
                        f"{prefix}/success_rate": float(metrics["success_rate"]),
                        f"{prefix}/mean_return": float(metrics["mean_return"]),
                        f"{prefix}/mean_steps": float(metrics["mean_steps"]),
                    }, step=effective)

                example_env = make_env(config_path, family)
                features, _ = example_env.reset(seed=int(seed))
                example_env.close()
                input_example = np.asarray([features], dtype=np.float32)
                model_info = mlflow.pyfunc.log_model(
                    name="pyfunc_model",
                    python_model=PPOPyFuncModel(),
                    artifacts={"model": str(model_path), "normalization": str(normalization_path)},
                    code_paths=[str(PROJECT_ROOT / "agents"), str(PROJECT_ROOT / "envs")],
                    pip_requirements=[
                        f"mlflow=={mlflow.__version__}", f"numpy=={np.__version__}", f"torch=={torch.__version__}",
                        f"stable-baselines3=={version('stable-baselines3')}", f"gymnasium=={version('gymnasium')}",
                        f"PyYAML=={version('PyYAML')}",
                    ],
                    input_example=input_example,
                    signature=infer_signature(input_example, np.zeros((1,), dtype=np.int64)),
                )
                registered_name = f"DroneRescuePPO-{family}-{level_id}"
                registered = mlflow.register_model(model_info.model_uri, registered_name)
                registry_uri = f"models:/{registered_name}/{registered.version}"

                package = Path(package_dir) / run_id
                if package.exists():
                    raise FileExistsError(f"El paquete del modelo ya existe: {package}")
                package.mkdir(parents=True)
                for source in (model_path, final_path, config_artifact, normalization_path, evaluation_path):
                    shutil.copyfile(source, package / source.name)

                for name, folder in (("model.zip", "checkpoints"), ("final.zip", "checkpoints"), ("config.json", "config"),
                                     ("normalization.json", "normalization"), ("evaluacion.json", "evaluation")):
                    mlflow.log_artifact(str(package / name), artifact_path=folder)

                manifest.update({
                    "config_uri": f"runs:/{run_id}/config/config.json",
                    "model_uri": f"runs:/{run_id}/checkpoints/model.zip",
                    "model_sha256": _sha256(package / "model.zip"),
                    "final_checkpoint_uri": f"runs:/{run_id}/checkpoints/final.zip",
                    "final_checkpoint_sha256": _sha256(package / "final.zip"),
                    "normalization_uri": f"runs:/{run_id}/normalization/normalization.json",
                    "normalization_sha256": _sha256(package / "normalization.json"),
                    "evaluation_uri": f"runs:/{run_id}/evaluation/evaluacion.json",
                    "evaluation_sha256": _sha256(package / "evaluacion.json"),
                    "package_path": str(package.relative_to(PROJECT_ROOT)) if package.is_relative_to(PROJECT_ROOT) else str(package),
                    "registry_uri": registry_uri,
                    "registry_name": registered_name,
                    "registry_version": str(registered.version),
                    "effective_timesteps": effective,
                    "training_duration_s": round(time.time() - started, 1),
                    "diagnostic": None,
                })
                manifest_path = _write_manifest(Path(manifest_dir), run_id, manifest)
                mlflow.log_artifact(str(manifest_path), artifact_path="manifest")
                mlflow.set_tags({"checkpoint_status": "disponible", "registry_uri": registry_uri,
                                 "config_sha256": config_sha256,
                                 "checkpoint_artifact_path": "checkpoints/final.zip"})
                return {"run_id": run_id, "manifest_path": str(manifest_path), "package": str(package),
                        "registry_uri": registry_uri, "evaluation": evaluation,
                        "best_during_training": callback.best_metrics}
        except Exception as error:
            if vector_env is not None:
                vector_env.close()
            detail = f"{type(error).__name__}: {error}"
            mlflow.set_tags({"checkpoint_status": "sin checkpoint disponible",
                             "diagnostic": "sin checkpoint disponible", "failure_detail": detail[:500]})
            mlflow.log_text(f"sin checkpoint disponible\n{detail}\n", "diagnostic.txt")
            manifest["failure_detail"] = detail
            try:
                _write_manifest(Path(manifest_dir), run_id, manifest)
            except FileExistsError:
                pass
            raise


def _make_worker(config_path: str, family: str, seed: int):
    env = make_env(config_path, family, fast=True)
    env.reset(seed=seed)
    return env


def _parent_checkpoint(parent_run_id: str, level_id: str, family: str) -> Path:
    parent = mlflow.get_run(parent_run_id)
    if parent.info.status != "FINISHED":
        raise ValueError(f"El run padre {parent_run_id} no terminó correctamente")
    tags = parent.data.tags
    if tags.get("level_id") != level_id or tags.get("model_family") != family:
        raise ValueError("Solo se puede continuar un run del mismo nivel y familia")
    return Path(mlflow.artifacts.download_artifacts(run_id=parent_run_id, artifact_path=tags["checkpoint_artifact_path"]))


def load_ppo_profile(profile_path: str | Path) -> tuple[dict, Path]:
    path = Path(profile_path).resolve()
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section in ("experiment", "level", "training", "model_family"):
        if not (profile or {}).get(section):
            raise ValueError(f"Falta {section} en el perfil")
    for key in ("id", "hypothesis"):
        if not profile["experiment"].get(key):
            raise ValueError(f"Falta experiment.{key} en el perfil")
    training = profile["training"]
    for key in ("timesteps", "n_envs", "evaluation_every", "evaluation_seeds", "agent"):
        if not training.get(key):
            raise ValueError(f"Falta training.{key} en el perfil")
    return profile, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parent-run-id")
    parser.add_argument("--timesteps", type=int, help="Sobrescribe el presupuesto del perfil; queda en el manifiesto")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))
    parser.add_argument("--allow-dirty", action="store_true", help="Solo pruebas: marca el run con git_dirty=true")
    parser.add_argument("--status-file", help="Escribe aquí el último progreso (lo lee la API)")
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE_DIR))
    args = parser.parse_args()
    configure_tracking()
    result = train_run(args.profile, args.seed, parent_run_id=args.parent_run_id, suite_path=args.suite,
                       manifest_dir=args.manifest_dir, package_dir=args.package_dir,
                       timesteps_override=args.timesteps, allow_dirty=args.allow_dirty, status_path=args.status_file)
    print(json.dumps({key: value for key, value in result.items() if key != "evaluation"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
