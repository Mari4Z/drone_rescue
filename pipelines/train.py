"""Entrenamiento reproducible de Q-learning con MLflow y manifiestos."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import yaml
from mlflow.models import infer_signature

from agents.qlearning import QAgent
from agents.qtable_model import QTablePyFuncModel
from envs.drone_env import ENVIRONMENT_VERSION, DroneEnv, discretize_observation, load_config
from pipelines.evaluate import (
    DEFAULT_SUITE_PATH,
    QLearningPolicy,
    evaluate,
    load_q_table,
    load_suite,
    save_q_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "manifests"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "q_tables"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def load_profile(profile_path: str | Path) -> tuple[dict, Path]:
    path = Path(profile_path).resolve()
    with path.open(encoding="utf-8") as profile_file:
        profile = yaml.safe_load(profile_file)
    required = {"experiment", "level", "training"}
    missing = required.difference(profile or {})
    if missing:
        raise ValueError(f"Faltan secciones en el perfil: {sorted(missing)}")
    for key in ("id", "hypothesis"):
        if not profile["experiment"].get(key):
            raise ValueError(f"Falta experiment.{key} en el perfil")
    if not profile["level"].get("id") or not profile["level"].get("config"):
        raise ValueError("El perfil debe definir level.id y level.config")
    training = profile["training"]
    if int(training.get("episodes", 0)) <= 0 or int(training.get("evaluation_interval", 0)) <= 0:
        raise ValueError("episodes y evaluation_interval deben ser mayores que cero")
    if not training.get("evaluation_seeds"):
        raise ValueError("El perfil debe registrar evaluation_seeds")
    return profile, path


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _evaluate_greedy(config_path: Path, q_table: dict, seeds: list[int]) -> list[dict]:
    outcomes = []
    for seed in seeds:
        environment = DroneEnv(config_path=config_path)
        observation, _ = environment.reset(seed=int(seed))
        policy = QLearningPolicy(q_table)
        total_return = 0.0
        terminated = truncated = False
        info = {"cause": None}
        while not (terminated or truncated):
            action = policy.act(observation, environment.config)
            observation, reward, terminated, truncated, info = environment.step(action)
            total_return += float(reward)
        outcomes.append({
            "seed": int(seed),
            "success": info["cause"] == "success",
            "cause": info["cause"],
            "steps": int(environment.steps),
            "return": total_return,
            "battery_remaining_wh": float(environment.battery_wh),
            "collisions": int(environment.contacts),
        })
        environment.close()
    return outcomes


def _resume_agent(
    parent_run_id: str,
    level_id: str,
    environment_config: dict,
    training_config: dict,
    tracking_uri: str | None,
) -> tuple[QAgent, float]:
    parent = mlflow.get_run(parent_run_id)
    if parent.info.status != "FINISHED":
        raise ValueError(f"El run padre {parent_run_id} no terminó correctamente")
    tags = parent.data.tags
    if tags.get("environment_version") != ENVIRONMENT_VERSION:
        raise ValueError("No se puede reanudar desde otra environment_version")
    if tags.get("level_id") != level_id:
        raise ValueError("No se puede reanudar desde otro nivel")
    environment_sha256 = hashlib.sha256(_json_bytes(environment_config)).hexdigest()
    if tags.get("environment_config_sha256") != environment_sha256:
        raise ValueError("No se puede reanudar con otra configuración resuelta del ambiente")
    checkpoint_artifact = tags.get("checkpoint_artifact_path")
    if not checkpoint_artifact:
        raise ValueError("El run padre no registra la ruta de su checkpoint")
    artifact = mlflow.artifacts.download_artifacts(
        run_id=parent_run_id,
        artifact_path=checkpoint_artifact,
        tracking_uri=tracking_uri,
    )
    table = load_q_table(artifact)
    params = parent.data.params
    agent_cfg = training_config["agent"]
    final_epsilon = parent.data.metrics.get("training/final_epsilon", float(agent_cfg["epsilon_start"]))
    agent = QAgent(
        n_actions=7,
        alpha=float(agent_cfg["alpha"]),
        gamma=float(agent_cfg["gamma"]),
        eps=float(final_epsilon),
        eps_min=float(agent_cfg["epsilon_min"]),
        eps_decay=float(agent_cfg["epsilon_decay"]),
        seed=int(params.get("seed", 0)),
    )
    for state, values in table.items():
        agent.q[state] = values.copy()
    return agent, agent.eps


def _write_manifest(directory: Path, run_id: str, manifest: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.json"
    if path.exists():
        raise FileExistsError(f"El manifiesto ya existe: {path}")
    path.write_bytes(_json_bytes(manifest))
    return path


def train_run(
    profile_path: str | Path,
    seed: int,
    *,
    parent_run_id: str | None = None,
    manifest_dir: str | Path = DEFAULT_MANIFEST_DIR,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    suite_path: str | Path | None = None,
    episodes_override: int | None = None,
) -> dict:
    profile, profile_file = load_profile(profile_path)
    experiment_id = str(profile["experiment"]["id"])
    level_id = str(profile["level"]["id"])
    hypothesis = str(profile["experiment"]["hypothesis"])
    parent_run_id = parent_run_id or profile.get("parent_run_id")
    training = dict(profile["training"])
    if episodes_override is not None:
        training["episodes"] = int(episodes_override)
    config_path = _project_path(profile["level"]["config"])
    suite_file = Path(suite_path).resolve() if suite_path else DEFAULT_SUITE_PATH
    suite, _ = load_suite(suite_file)
    git_commit = _git_commit()
    mlflow.set_experiment(str(profile.get("mlflow_experiment", "drone-rescue-training")))
    tracking_uri = mlflow.get_tracking_uri()

    run_name = f"{experiment_id}-seed-{int(seed)}"
    tags = {
        "git_commit": git_commit,
        "environment_version": ENVIRONMENT_VERSION,
        "level_id": level_id,
        "hypothesis": hypothesis,
        "parent_run_id": str(parent_run_id or "none"),
        "profile_id": str(profile.get("profile_id", profile_file.stem)),
        "evaluation_suite_id": str(suite["evaluation_suite_id"]),
        "checkpoint_status": "sin checkpoint disponible",
    }
    manifest_base = {
        "run_id": None,
        "level_id": level_id,
        "environment_version": ENVIRONMENT_VERSION,
        "git_commit": git_commit,
        "config_sha256": None,
        "model_uri": None,
        "model_sha256": None,
        "normalization_uri": None,
        "normalization_sha256": None,
        "runtime_ref": (
            f"python=={platform.python_version()}; numpy=={np.__version__}; "
            f"gymnasium=={version('gymnasium')}; PyYAML=={version('PyYAML')}; mlflow=={mlflow.__version__}"
        ),
        "evaluation_suite_id": str(suite["evaluation_suite_id"]),
        "evaluation_suite_version": str(suite["version"]),
        "evaluation_suite_sha256": _sha256(suite_file),
        "seed": int(seed),
        "parent_run_id": parent_run_id,
        "diagnostic": "sin checkpoint disponible",
    }

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        manifest_base["run_id"] = run.info.run_id
        try:
            environment_config = load_config(config_path)
            if str(environment_config.get("level", {}).get("id", level_id)) != level_id:
                raise ValueError("level.id del perfil no coincide con la configuración YAML")
            resolved_snapshot = {
                "profile": profile,
                "profile_path": str(profile_file.relative_to(PROJECT_ROOT)) if profile_file.is_relative_to(PROJECT_ROOT) else str(profile_file),
                "environment": environment_config,
                "seed": int(seed),
                "parent_run_id": parent_run_id,
                "evaluation_suite_id": suite["evaluation_suite_id"],
            }
            config_bytes = _json_bytes(resolved_snapshot)
            config_sha256 = hashlib.sha256(config_bytes).hexdigest()
            environment_config_sha256 = hashlib.sha256(_json_bytes(environment_config)).hexdigest()
            manifest_base["config_sha256"] = config_sha256
            agent_cfg = training["agent"]
            mlflow.log_params({
                "experiment_id": experiment_id,
                "profile_id": str(profile.get("profile_id", profile_file.stem)),
                "level_config": str(profile["level"]["config"]),
                "level_id": level_id,
                "environment_version": ENVIRONMENT_VERSION,
                "reward_version": str(environment_config["reward"]["version"]),
                "model_family": str(environment_config.get("level", {}).get("model_family", "drone-rescue-v2")),
                "parent_run_id": str(parent_run_id or "none"),
                "seed": int(seed),
                "episodes": int(training["episodes"]),
                "evaluation_interval": int(training["evaluation_interval"]),
                "evaluation_seeds": json.dumps([int(value) for value in training["evaluation_seeds"]]),
                "alpha": float(agent_cfg["alpha"]),
                "gamma": float(agent_cfg["gamma"]),
                "epsilon_start": float(agent_cfg["epsilon_start"]),
                "epsilon_min": float(agent_cfg["epsilon_min"]),
                "epsilon_decay": float(agent_cfg["epsilon_decay"]),
                "config_sha256": config_sha256,
            })
            mlflow.set_tags({
                **tags,
                "config_sha256": config_sha256,
                "environment_config_sha256": environment_config_sha256,
            })

            if parent_run_id:
                agent, initial_epsilon = _resume_agent(
                    parent_run_id, level_id, environment_config, training, tracking_uri
                )
            else:
                agent = QAgent(
                    n_actions=7,
                    alpha=float(agent_cfg["alpha"]),
                    gamma=float(agent_cfg["gamma"]),
                    eps=float(agent_cfg["epsilon_start"]),
                    eps_min=float(agent_cfg["epsilon_min"]),
                    eps_decay=float(agent_cfg["epsilon_decay"]),
                    seed=int(seed),
                )
                initial_epsilon = agent.eps
            if parent_run_id:
                mlflow.set_tag("parent_run_id", parent_run_id)
                mlflow.log_param("parent_final_epsilon", initial_epsilon)

            environment = DroneEnv(config_path=config_path)
            training_curve = []
            evaluation_curve = []
            eval_seeds = [int(value) for value in training["evaluation_seeds"]]
            total_episodes = int(training["episodes"])

            for episode_index in range(1, total_episodes + 1):
                observation, _ = environment.reset(seed=int(seed) if episode_index == 1 else None)
                state = discretize_observation(observation, environment.config)
                terminated = truncated = False
                episode_return = 0.0
                info = {"cause": None}
                while not (terminated or truncated):
                    action = agent.act(state)
                    next_observation, reward, terminated, truncated, info = environment.step(action)
                    next_state = discretize_observation(next_observation, environment.config)
                    agent.update(state, action, reward, next_state, terminated)
                    state = next_state
                    episode_return += float(reward)
                agent.decay()
                training_metrics = {
                    "episode": episode_index,
                    "return": episode_return,
                    "success": info["cause"] == "success",
                    "epsilon": float(agent.eps),
                    "cause": info["cause"],
                    "steps": int(environment.steps),
                }
                training_curve.append(training_metrics)
                mlflow.log_metrics({
                    "training/return": episode_return,
                    "training/success": float(training_metrics["success"]),
                    "training/epsilon": float(agent.eps),
                }, step=episode_index)

                if episode_index % int(training["evaluation_interval"]) == 0 or episode_index == total_episodes:
                    current_q_table = {state_key: value.copy() for state_key, value in agent.q.items()}
                    greedy_episodes = _evaluate_greedy(config_path, current_q_table, eval_seeds)
                    eval_success = float(np.mean([item["success"] for item in greedy_episodes]))
                    eval_return = float(np.mean([item["return"] for item in greedy_episodes]))
                    eval_record = {
                        "episode": episode_index,
                        "epsilon": 0.0,
                        "success_rate": eval_success,
                        "mean_return": eval_return,
                        "episodes": greedy_episodes,
                    }
                    evaluation_curve.append(eval_record)
                    mlflow.log_metrics({
                        "evaluation/return": eval_return,
                        "evaluation/success": eval_success,
                        "evaluation/epsilon": 0.0,
                    }, step=episode_index)
                    for eval_episode in greedy_episodes:
                        seed_metric_suffix = f"seed_{eval_episode['seed']}"
                        mlflow.log_metrics({
                            f"evaluation/episode_return/{seed_metric_suffix}": float(eval_episode["return"]),
                            f"evaluation/episode_success/{seed_metric_suffix}": float(eval_episode["success"]),
                            f"evaluation/episode_epsilon/{seed_metric_suffix}": 0.0,
                        }, step=episode_index)

            environment.close()
            normalization = {
                "method": "discretize_observation",
                "family": "drone-rescue-v2",
                "parameters": environment_config["q_learning"]["discretization"],
                "state_dimensions": 16,
                "action_dimensions": 7,
            }

            with tempfile.TemporaryDirectory(prefix=f"drone-rescue-{run.info.run_id}-") as tmp:
                temp_dir = Path(tmp)
                checkpoint_path = save_q_table(agent, temp_dir / "q_table.npy")
                config_artifact = temp_dir / "config.json"
                config_artifact.write_bytes(config_bytes)
                normalization_artifact = temp_dir / "normalization.json"
                normalization_artifact.write_bytes(_json_bytes(normalization))
                normalization_sha256 = _sha256(normalization_artifact)
                evaluation_path = temp_dir / "evaluacion.json"
                final_evaluation = evaluate(suite_file, checkpoint_path, evaluation_path)
                final_evaluation["training_curve"] = training_curve
                final_evaluation["evaluation_curve"] = evaluation_curve
                evaluation_path.write_bytes(_json_bytes(final_evaluation))
                for case in final_evaluation["cases"]:
                    case_prefix = f"final_evaluation/{case['case_id']}"
                    mlflow.log_metrics({
                        f"{case_prefix}/success_rate": float(case["metrics"]["success_rate"]),
                        f"{case_prefix}/mean_return": float(case["metrics"]["mean_return"]),
                    }, step=total_episodes)

                obs_example, _ = environment.reset(seed=int(seed))
                input_example = np.asarray([obs_example], dtype=np.float32)
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="pyfunc_model",
                    python_model=QTablePyFuncModel(),
                    artifacts={
                        "q_table": str(checkpoint_path),
                        "config": str(config_artifact),
                        "normalization": str(normalization_artifact),
                    },
                    code_paths=[str(PROJECT_ROOT / "agents"), str(PROJECT_ROOT / "envs")],
                    pip_requirements=[
                        f"mlflow=={mlflow.__version__}",
                        f"numpy=={np.__version__}",
                        "gymnasium==1.4.0",
                        "PyYAML==6.0.3",
                    ],
                    input_example=input_example,
                    signature=infer_signature(input_example, np.zeros((1,), dtype=np.int64)),
                )
                registered_name = f"DroneRescueQTable-{level_id}"
                registered_version = mlflow.register_model(model_info.model_uri, registered_name)
                registry_uri = f"models:/{registered_name}/{registered_version.version}"

                final_checkpoint_dir = Path(checkpoint_dir)
                if not final_checkpoint_dir.is_absolute():
                    final_checkpoint_dir = PROJECT_ROOT / final_checkpoint_dir
                final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                final_checkpoint = final_checkpoint_dir / f"{run.info.run_id}.npy"
                if final_checkpoint.exists():
                    raise FileExistsError(f"El checkpoint ya existe: {final_checkpoint}")
                shutil.copyfile(checkpoint_path, final_checkpoint)
                mlflow.log_artifact(str(final_checkpoint), artifact_path="checkpoints")
                mlflow.log_artifact(str(config_artifact), artifact_path="config")
                mlflow.log_artifact(str(normalization_artifact), artifact_path="normalization")
                mlflow.log_artifact(str(evaluation_path), artifact_path="evaluation")

                checkpoint_sha256 = _sha256(final_checkpoint)
                manifest_base.update({
                    "model_uri": f"runs:/{run.info.run_id}/checkpoints/{final_checkpoint.name}",
                    "model_sha256": checkpoint_sha256,
                    "normalization_uri": f"runs:/{run.info.run_id}/normalization/normalization.json",
                    "normalization_sha256": normalization_sha256,
                    "config_uri": f"runs:/{run.info.run_id}/config/config.json",
                    "registry_uri": registry_uri,
                    "registry_name": registered_name,
                    "registry_version": str(registered_version.version),
                    "evaluation_sha256": _sha256(evaluation_path),
                    "diagnostic": None,
                })
                manifest_path = _write_manifest(Path(manifest_dir), run.info.run_id, manifest_base)
                mlflow.log_artifact(str(manifest_path), artifact_path="manifest")
                mlflow.set_tags({
                    "checkpoint_status": "disponible",
                    "registry_uri": registry_uri,
                    "config_sha256": config_sha256,
                    "checkpoint_artifact_path": f"checkpoints/{final_checkpoint.name}",
                })
                mlflow.log_metric("training/final_epsilon", float(agent.eps), step=total_episodes)
                return {
                    "run_id": run.info.run_id,
                    "run_name": run_name,
                    "manifest_path": str(manifest_path),
                    "checkpoint_path": str(final_checkpoint),
                    "registry_uri": registry_uri,
                    "evaluation": final_evaluation,
                }
        except Exception as error:
            diagnostic = "sin checkpoint disponible"
            mlflow.set_tags({
                "checkpoint_status": "sin checkpoint disponible",
                "diagnostic": diagnostic,
                "failure_detail": f"{type(error).__name__}: {error}"[:500],
            })
            mlflow.log_text(f"{diagnostic}\n{type(error).__name__}: {error}\n", "diagnostic.txt")
            manifest_base["diagnostic"] = diagnostic
            manifest_base["failure_detail"] = f"{type(error).__name__}: {error}"
            try:
                _write_manifest(Path(manifest_dir), run.info.run_id, manifest_base)
            except FileExistsError:
                pass
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Perfil YAML de configs/training")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--parent-run-id", help="Reanuda desde el checkpoint de un run FINISHED")
    parser.add_argument("--episodes", type=int, help="Sobrescribe el presupuesto del perfil para esta ejecución")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH), help="Suite pública fija para evaluación final")
    args = parser.parse_args()
    result = train_run(
        args.profile,
        args.seed,
        parent_run_id=args.parent_run_id,
        suite_path=args.suite,
        episodes_override=args.episodes,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "evaluation"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
