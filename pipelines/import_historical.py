"""Registra en MLflow modelos anteriores a este pipeline, con su procedencia declarada.

No sustituyen a un run reproducible: conservan el intento, su hash y el commit más cercano al
entrenamiento, y quedan marcados con provenance=historico.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import mlflow
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym

import envs.grid_drone_env as grid
from agents.dqn import DQNAgent as ContinuousDQN, DQNPolicy
from agents.grid.dqn import DQNAgent
from agents.grid.evaluate import run_episode
from agents.grid.qlearning import QAgent
from envs.drone_env import DroneEnv
from pipelines.tracking import configure_tracking
from pipelines.train import _json_bytes, _write_manifest
from pipelines.train_ppo import _sha256

SUITE_PATH = PROJECT_ROOT / "configs" / "grid_evaluation_suite.yaml"
STATUS = "histórico: se recupera con cli.py o con el paquete, no por la API"

MODELS = [
    {"id": "grid-qtable-dummy", "file": "models/dummy.npy", "kind": "qtable", "env": "DroneRescueGridDummy-v1", "commit": "1eed155",
     "hypothesis": "Q-learning tabular aprende a llegar de A a B con una sola columna fija."},
    {"id": "grid-qtable-5col", "file": "models/v0.npy", "kind": "qtable", "env": "DroneRescueGrid-v0", "commit": "4587714",
     "hypothesis": "Q-learning tabular aprende a llegar de A a B con cinco columnas fijas."},
    {"id": "grid-dqn-sin-memoria", "file": "models/dqn_sin_memoria.pt", "kind": "dqn-39", "env": "DroneRescueGridRandom-v2", "commit": "425f4a7",
     "hypothesis": "Un DQN con sensores generaliza a mapas aleatorios de 5 a 25 columnas.",
     "note": "Espera 39 observaciones; se evalúa con las 39 primeras del ambiente actual, que no cambiaron."},
    {"id": "dqn-v2-n1-exploratorio", "file": "models/dqn_n1_s0.pt", "kind": "dqn-v2", "env": "configs/levels/n1_mision_base.yaml", "commit": "ba3bcd7",
     "family": "drone-rescue-v2", "hypothesis": "Un DQN con entradas de escala fina aprende la misión base con control directo de fuerzas.",
     "note": "Run exploratorio, sin MLflow al entrenar. Rechazado: no entrega."},
    {"id": "grid-dqn-memoria", "file": "models/dqn.pt", "kind": "dqn-46", "env": "DroneRescueGridRandom-v2", "commit": "58ff684",
     "hypothesis": "La memoria de visitas evita que el DQN oscile entre dos celdas en mapas aleatorios."},
]


def _policy(spec: dict):
    if spec["kind"] == "qtable":
        agent = QAgent((grid.NX, grid.NY, grid.NZ), 6)
        agent.q = np.load(PROJECT_ROOT / spec["file"])
        return lambda obs: agent.act(tuple(obs), greedy=True)
    agent = DQNAgent.load(PROJECT_ROOT / spec["file"], "cpu")
    if spec["kind"] == "dqn-39":
        return lambda obs: agent.act(obs[:39])
    return agent.act


def _evaluate_v2(spec: dict) -> dict:
    agent = DQNPolicy(ContinuousDQN.load(PROJECT_ROOT / spec["file"]))
    env = DroneEnv(PROJECT_ROOT / spec["env"])
    episodes = []
    for seed in (101, 211, 307, 401, 503):
        observation, _ = env.reset(seed=seed)
        done, info = False, {}
        while not done:
            observation, _, terminated, truncated, info = env.step(agent.act(observation))
            done = terminated or truncated
        episodes.append({"seed": seed, "result": info["result"], "cause": info["cause"], "stage": info["stage"], "steps": env.steps})
    goals = sum(episode["result"] == "goal" for episode in episodes)
    return {"environment": spec["env"], "episodes": episodes,
            "metrics": {"success_rate": goals / len(episodes), "episodes": len(episodes),
                        "mean_steps": float(np.mean([episode["steps"] for episode in episodes]))}}


def _evaluate(spec: dict, suite: dict) -> dict:
    if spec["kind"] == "dqn-v2":
        return _evaluate_v2(spec)
    env = gym.make(spec["env"])
    random_map = getattr(env.unwrapped, "random_map", False)
    count = int(suite["episodes_per_map_set"]) if random_map else 1
    policy = _policy(spec)
    episodes = []
    for index in range(count):
        seed = int(suite["first_seed"]) + index
        path, result = run_episode(env, policy, seed=seed)
        episodes.append({"seed": seed, "result": result, "steps": len(path) - 1,
                         "columns": len(env.unwrapped.columns)})
    goals = sum(episode["result"] == "goal" for episode in episodes)
    return {"evaluation_suite_id": suite["evaluation_suite_id"], "evaluation_suite_version": suite["version"],
            "environment": spec["env"], "episodes": episodes,
            "metrics": {"success_rate": goals / count, "episodes": count,
                        "mean_steps": float(np.mean([episode["steps"] for episode in episodes]))}}


def import_model(spec: dict, suite: dict, manifest_dir: Path, package_dir: Path) -> None:
    source = PROJECT_ROOT / spec["file"]
    mlflow.set_experiment("drone-rescue-training")
    tags = {"provenance": "historico", "git_commit": spec["commit"], "model_family": spec.get("family", grid.FAMILY), "algorithm": spec["kind"],
            "hypothesis": spec["hypothesis"], "level_id": spec["env"], "environment_version": grid.ENVIRONMENT_VERSION,
            "checkpoint_status": STATUS, "evaluation_suite_id": suite["evaluation_suite_id"]}
    with mlflow.start_run(run_name=spec["id"], tags=tags) as run:
        run_id = run.info.run_id
        evaluation = _evaluate(spec, suite)
        package = package_dir / run_id
        package.mkdir(parents=True)
        shutil.copyfile(source, package / source.name)
        contract = {"family": spec.get("family", grid.FAMILY), "kind": spec["kind"], "environment": spec["env"], "note": spec.get("note"),
                    "actions": ["+x", "-x", "+y", "-y", "subir", "bajar"], "contract": "docs/familia_cuadricula.md"}
        (package / "normalization.json").write_bytes(_json_bytes(contract))
        (package / "evaluacion.json").write_bytes(_json_bytes(evaluation))
        mlflow.log_params({"experiment_id": spec["id"], "environment": spec["env"], "reward_version": grid.REWARD_VERSION,
                           "episodes_evaluated": evaluation["metrics"]["episodes"]})
        mlflow.log_metrics({"evaluation/success": evaluation["metrics"]["success_rate"], "evaluation/steps": evaluation["metrics"]["mean_steps"]})
        for name, folder in ((source.name, "checkpoints"), ("normalization.json", "normalization"), ("evaluacion.json", "evaluation")):
            mlflow.log_artifact(str(package / name), artifact_path=folder)
        manifest = {
            "run_id": run_id, "provenance": "historico", "level_id": spec["env"], "experiment_id": spec["id"],
            "algorithm": spec["kind"], "model_family": spec.get("family", grid.FAMILY), "environment_version": grid.ENVIRONMENT_VERSION,
            "reward_version": grid.REWARD_VERSION, "git_commit": spec["commit"],
            "git_commit_note": "commit más cercano al entrenamiento; el modelo se entrenó antes de este registro",
            "model_uri": f"runs:/{run_id}/checkpoints/{source.name}", "model_sha256": _sha256(source),
            "normalization_uri": f"runs:/{run_id}/normalization/normalization.json",
            "normalization_sha256": _sha256(package / "normalization.json"),
            "evaluation_uri": f"runs:/{run_id}/evaluation/evaluacion.json", "evaluation_sha256": _sha256(package / "evaluacion.json"),
            "evaluation_suite_id": suite["evaluation_suite_id"], "evaluation_suite_version": suite["version"],
            "evaluation_suite_sha256": _sha256(SUITE_PATH), "package_path": str(package.relative_to(PROJECT_ROOT)),
            "seed": None, "parent_run_id": None, "diagnostic": None, "serving": STATUS,
            "recovery": f"python cli.py eval --model {spec['file']} --env {spec['env']}",
        }
        path = _write_manifest(manifest_dir, run_id, manifest)
        mlflow.log_artifact(str(path), artifact_path="manifest")
        print(spec["id"], run_id, evaluation["metrics"], flush=True)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    configure_tracking()
    suite = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    only = set(sys.argv[1:])
    for spec in [model for model in MODELS if not only or model["id"] in only]:
        import_model(spec, suite, PROJECT_ROOT / "manifests", PROJECT_ROOT / "artifacts" / "models")


if __name__ == "__main__":
    main()
