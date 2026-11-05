"""Evaluación reproducible de políticas para la suite pública Drone Rescue."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.qlearning import QAgent
from envs.drone_env import ENVIRONMENT_VERSION, DroneEnv, discretize_observation, load_config

DEFAULT_SUITE_PATH = PROJECT_ROOT / "configs" / "evaluation_suite.yaml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "results" / "evaluacion.json"
REFERENCE_IDS = ("random", "rule", "q_learning_greedy")


def load_suite(suite_path: str | Path = DEFAULT_SUITE_PATH) -> tuple[dict, Path]:
    path = Path(suite_path).resolve()
    with path.open(encoding="utf-8") as suite_file:
        suite = yaml.safe_load(suite_file)
    required = {"evaluation_suite_id", "version", "split", "episodes_per_level", "seeds", "levels", "references"}
    missing = required.difference(suite or {})
    if missing:
        raise ValueError(f"Faltan campos de suite: {sorted(missing)}")
    if suite["split"] != "public_validation":
        raise ValueError("Este evaluador solo acepta suites public_validation; el test reservado no se versiona aquí")
    if len(suite["seeds"]) != int(suite["episodes_per_level"]):
        raise ValueError("La suite fija requiere una semilla registrada por episodio y nivel")
    if tuple(suite["references"]) != REFERENCE_IDS:
        raise ValueError(f"Las referencias de suite deben ser exactamente {REFERENCE_IDS}")
    if len({int(seed) for seed in suite["seeds"]}) != len(suite["seeds"]):
        raise ValueError("Las semillas de evaluación deben ser únicas")
    return suite, path


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_q_table(path: str | Path) -> dict[tuple[int, ...], np.ndarray]:
    """Lee una tabla sparse NPZ o NPY estructurada sin objetos/pickle."""
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        if loaded.dtype.names != ("states", "q_values"):
            raise ValueError("El checkpoint NPY debe tener campos states[16] y q_values[7]")
        states = np.asarray(loaded["states"], dtype=np.int64)
        q_values = np.asarray(loaded["q_values"], dtype=np.float64)
    else:
        with loaded as data:
            if not {"states", "q_values"}.issubset(data.files):
                raise ValueError("El checkpoint NPZ debe contener los arrays states y q_values")
            states = np.asarray(data["states"], dtype=np.int64)
            q_values = np.asarray(data["q_values"], dtype=np.float64)
    if states.ndim != 2 or q_values.ndim != 2 or states.shape[0] != q_values.shape[0]:
        raise ValueError("states y q_values deben ser matrices con la misma cantidad de filas")
    if q_values.shape[1] != 7 or states.shape[1] != 16:
        raise ValueError("La tabla no corresponde a drone-rescue-v2 (16 bins, 7 acciones)")
    if not np.isfinite(q_values).all():
        raise ValueError("La tabla Q contiene valores no finitos")
    return {tuple(state.tolist()): values for state, values in zip(states, q_values)}


def save_q_table(agent: QAgent, path: str | Path) -> Path:
    """Exporta una tabla sparse como NPY estructurado o NPZ según extensión."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"El checkpoint ya existe: {target}. Usa una ruta/version nueva.")
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted((tuple(map(int, state)), np.asarray(values, dtype=np.float64)) for state, values in agent.q.items())
    states = np.asarray([state for state, _ in entries], dtype=np.int64).reshape((-1, 16))
    q_values = np.asarray([values for _, values in entries], dtype=np.float64).reshape((-1, 7))
    if target.suffix == ".npy":
        table = np.empty(len(entries), dtype=[("states", np.int64, (16,)), ("q_values", np.float64, (7,))])
        table["states"] = states
        table["q_values"] = q_values
        np.save(target, table, allow_pickle=False)
    else:
        np.savez_compressed(target, states=states, q_values=q_values)
    return target


def evaluate_from_manifest(manifest_path: str | Path, suite_path: str | Path, output_path: str | Path) -> dict:
    """Reproduce la suite usando exactamente el checkpoint declarado por manifiesto."""
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    required = {
        "model_uri", "model_sha256", "config_uri", "config_sha256",
        "normalization_uri", "normalization_sha256", "evaluation_suite_id",
        "evaluation_suite_sha256", "environment_version",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Faltan campos en manifiesto: {sorted(missing)}")
    suite, _ = load_suite(suite_path)
    if suite["evaluation_suite_id"] != manifest["evaluation_suite_id"]:
        raise ValueError("El id de suite no coincide con el manifiesto")
    if manifest.get("evaluation_suite_version", suite["version"]) != suite["version"]:
        raise ValueError("La versión de suite no coincide con el manifiesto")
    suite_sha256 = hashlib.sha256(Path(suite_path).resolve().read_bytes()).hexdigest()
    if suite_sha256 != manifest["evaluation_suite_sha256"]:
        raise ValueError("SHA-256 de la suite no coincide con el manifiesto")
    if manifest["environment_version"] != ENVIRONMENT_VERSION:
        raise ValueError("La environment_version actual no coincide con el manifiesto")
    import mlflow

    checkpoint_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=manifest["model_uri"]))
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_sha256 != manifest["model_sha256"]:
        raise ValueError("SHA-256 del checkpoint no coincide con el manifiesto")
    config_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=manifest["config_uri"]))
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if config_sha256 != manifest["config_sha256"]:
        raise ValueError("SHA-256 de config.json no coincide con el manifiesto")
    normalization_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=manifest["normalization_uri"]))
    normalization_sha256 = hashlib.sha256(normalization_path.read_bytes()).hexdigest()
    if normalization_sha256 != manifest["normalization_sha256"]:
        raise ValueError("SHA-256 de la discretización no coincide con el manifiesto")
    return evaluate(suite_path, checkpoint_path, output_path)


class RandomPolicy:
    def __init__(self, n_actions: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def act(self, observation: np.ndarray, config: dict) -> int:
        del observation, config
        return int(self.rng.integers(self.n_actions))


class RulePolicy:
    """Control reactivo: dirige al objetivo, evita rayos bloqueados y frena al llegar."""

    def act(self, observation: np.ndarray, config: dict) -> int:
        obs = np.asarray(observation, dtype=np.float64)
        world = config["world"]
        physics = config["physics"]
        directions = np.asarray(config["actions"]["movement_directions"], dtype=np.float64)
        stage = int(np.argmax(obs[15:]))
        has_supply = bool(obs[14] >= 0.5)
        velocity = obs[3:6]
        speed = float(np.linalg.norm(velocity))
        target = obs[0:3].copy()
        altitude = float(obs[6])

        if stage == 0:  # outbound: cualquier movimiento inicia recogida; subir es seguro en BASE.
            return 4

        if stage == 1 and has_supply:
            horizontal_error = float(np.linalg.norm(target[:2]))
            if altitude < float(world["delivery_min_altitude_m"]):
                target[2] = float(world["delivery_min_altitude_m"]) - altitude
            elif altitude > float(world["delivery_max_altitude_m"]):
                target[2] = float(world["delivery_max_altitude_m"]) - altitude
            elif horizontal_error <= float(world["delivery_radius_m"]):
                target[:2] = 0.0
            inside_zone = (
                horizontal_error <= float(world["delivery_radius_m"])
                and float(world["delivery_min_altitude_m"]) <= altitude <= float(world["delivery_max_altitude_m"])
            )
            if inside_zone and speed <= float(world["delivery_max_speed_mps"]):
                return int(config["actions"]["release_index"])
            speed_limit = float(world["delivery_max_speed_mps"]) if inside_zone else float(physics["max_speed_mps"])
        elif stage == 2:
            horizontal_error = float(np.linalg.norm(target[:2]))
            if horizontal_error <= float(world["base_radius_m"]):
                target[:2] = 0.0
                if altitude > float(world["base_max_altitude_m"]):
                    target[2] = float(world["base_max_altitude_m"]) - altitude
            speed_limit = float(world["base_max_speed_mps"])
        else:
            speed_limit = float(physics["max_speed_mps"])

        target_norm = float(np.linalg.norm(target))
        if target_norm < 1e-9:
            target_unit = np.zeros(3)
        else:
            target_unit = target / target_norm

        # Próxima velocidad aproximada con la misma fuerza y dt; el viento no se
        # observa exactamente, por lo que la regla usa sensores y corrige después.
        dt = float(physics["dt_s"])
        mass = float(physics["mass_kg"])
        acceleration = float(physics["control_force_n"]) / mass
        predicted_velocities = velocity[None, :] + directions * acceleration * dt
        predicted_speeds = np.linalg.norm(predicted_velocities, axis=1)
        max_speed = min(float(physics["max_speed_mps"]), max(speed_limit, 0.1))
        scale = np.minimum(1.0, max_speed / np.maximum(predicted_speeds, 1e-9))
        predicted_velocities *= scale[:, None]
        displacements = predicted_velocities * dt
        alignment = displacements @ target_unit
        desired_speed = min(float(physics["max_speed_mps"]), max(target_norm, speed_limit))
        speed_error = np.linalg.norm(predicted_velocities, axis=1) - desired_speed
        scores = alignment - 0.08 * np.abs(speed_error)

        sensors = obs[7:13]
        sensor_clearance = float(config["physics"]["drone_radius_m"]) + float(config["sensors"]["ray_step_m"])
        for action, reading in enumerate(sensors):
            if reading <= sensor_clearance:
                scores[action] -= 100.0
            elif reading < 3.0 * sensor_clearance:
                scores[action] -= (3.0 * sensor_clearance - reading) * 2.0

        # Cerca de la meta, desacelerar es prioritario frente a seguir acelerando.
        if target_norm <= max(float(world["base_radius_m"]), float(world["delivery_radius_m"])) and speed > speed_limit:
            dominant_axis = int(np.argmax(np.abs(velocity)))
            opposite = directions.tolist().index((-np.eye(3)[dominant_axis] * np.sign(velocity[dominant_axis])).tolist())
            scores[opposite] += 4.0
        return int(np.argmax(scores))


class QLearningPolicy:
    def __init__(self, q_table: dict[tuple[int, ...], np.ndarray]):
        self.q_table = q_table

    def act(self, observation: np.ndarray, config: dict) -> int:
        state = discretize_observation(observation, config)
        values = self.q_table.get(tuple(state))
        return 0 if values is None else int(np.argmax(values))


def _make_policy(reference: str, seed: int, q_table: dict[tuple[int, ...], np.ndarray] | None):
    if reference == "random":
        return RandomPolicy(7, seed)
    if reference == "rule":
        return RulePolicy()
    if reference == "q_learning_greedy":
        if q_table is None:
            raise ValueError("La referencia Q-learning requiere --q-table (checkpoint NPZ entrenado)")
        return QLearningPolicy(q_table)
    raise ValueError(f"Referencia desconocida: {reference}")


def _summarize_case(episodes: list[dict], reward_version: str) -> dict:
    cause_counts = Counter(episode["cause"] for episode in episodes)

    def mean(field: str):
        values = [episode[field] for episode in episodes if episode[field] is not None]
        return float(np.mean(values)) if values else None

    successful = [episode for episode in episodes if episode["success"]]
    return {
        "episodes": len(episodes),
        "reward_version": reward_version,
        "success_count": len(successful),
        "success_rate": len(successful) / len(episodes),
        "cause_counts": dict(sorted(cause_counts.items())),
        # Este promedio queda dentro del caso nivel+política+reward. No se agrega
        # a través de niveles ni de versiones de recompensa.
        "mean_return": mean("return"),
        "mean_steps": mean("steps"),
        "mean_battery_remaining_wh": mean("battery_remaining_wh"),
        "mean_delivery_precision_m": mean("delivery_precision_m"),
        "mean_delivery_speed_mps": mean("delivery_speed_mps"),
        "total_collisions": int(sum(episode["collisions"] for episode in episodes)),
        "mean_inference_latency_ms": mean("inference_latency_ms"),
        "mean_total_inference_latency_ms": mean("total_inference_latency_ms"),
    }


def evaluate(suite_path: str | Path, q_table_path: str | Path, output_path: str | Path) -> dict:
    suite, resolved_suite_path = load_suite(suite_path)
    q_table = load_q_table(q_table_path)
    cases = []
    episode_count = int(suite["episodes_per_level"])
    seeds = [int(seed) for seed in suite["seeds"]]

    for level in suite["levels"]:
        config_path = _resolve_project_path(level["config"])
        resolved_config = load_config(config_path)
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        reward_version = str(resolved_config["reward"]["version"])
        for reference in suite["references"]:
            episodes = []
            for seed in seeds:
                environment = DroneEnv(config_path=config_path)
                observation, _ = environment.reset(seed=seed)
                policy = _make_policy(reference, seed + 1_000_003, q_table)
                total_return = 0.0
                total_latency_s = 0.0
                latency_samples_s = []
                delivery_precision_m = None
                delivery_speed_mps = None
                terminated = truncated = False
                info = {"cause": None, "contacts": 0}

                while not (terminated or truncated):
                    before_stage = environment.stage
                    carrying = environment.has_supply
                    started = time.perf_counter()
                    action = policy.act(observation, environment.config)
                    elapsed = time.perf_counter() - started
                    latency_samples_s.append(elapsed)
                    total_latency_s += elapsed
                    if action == int(environment.config["actions"]["release_index"]) and before_stage == "delivery" and carrying and delivery_precision_m is None:
                        delivery_precision_m = float(np.linalg.norm((environment.position - environment._delivery)[:2]))
                        delivery_speed_mps = float(np.linalg.norm(environment.velocity))
                    observation, reward, terminated, truncated, info = environment.step(action)
                    total_return += float(reward)

                episodes.append({
                    "level_id": level["id"],
                    "policy_id": reference,
                    "seed": seed,
                    "reward_version": reward_version,
                    "success": info["cause"] == "success",
                    "cause": info["cause"],
                    "steps": int(environment.steps),
                    "battery_remaining_wh": float(environment.battery_wh),
                    "delivery_precision_m": delivery_precision_m,
                    "delivery_speed_mps": delivery_speed_mps,
                    "collisions": int(environment.contacts),
                    "return": total_return,
                    "inference_latency_ms": float(np.mean(latency_samples_s) * 1000.0) if latency_samples_s else 0.0,
                    "total_inference_latency_ms": total_latency_s * 1000.0,
                })
                environment.close()

            cases.append({
                "case_id": f"{level['id']}-{reference}",
                "level_id": level["id"],
                "config_path": str(level["config"]),
                "config_sha256": config_sha256,
                "reward_version": reward_version,
                "policy_id": reference,
                "episodes": episodes,
                "metrics": _summarize_case(episodes, reward_version),
            })

    result = {
        "evaluation_suite_id": suite["evaluation_suite_id"],
        "evaluation_suite_version": suite["version"],
        "split": suite["split"],
        "suite_config_path": str(resolved_suite_path.relative_to(PROJECT_ROOT)) if resolved_suite_path.is_relative_to(PROJECT_ROOT) else str(resolved_suite_path),
        "evaluation_suite_sha256": hashlib.sha256(resolved_suite_path.read_bytes()).hexdigest(),
        "environment_version": ENVIRONMENT_VERSION,
        "seeds": seeds,
        "episodes_per_level": episode_count,
        "reference_policies": list(suite["references"]),
        "q_table_sha256": hashlib.sha256(Path(q_table_path).read_bytes()).hexdigest(),
        "return_comparison_scope": "solo dentro del mismo nivel y reward_version; sin agregado entre casos",
        "cases": cases,
    }
    output = Path(output_path)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise FileExistsError(f"La salida ya existe: {output}. Usa una ruta/version nueva para conservar el resultado anterior.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH), help="Suite versionada (split public_validation)")
    parser.add_argument("--q-table", required=True, help="Q-table entrenada en formato NPZ states/q_values")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Ruta de salida de evaluacion.json")
    args = parser.parse_args()
    result = evaluate(args.suite, args.q_table, args.output)
    print(json.dumps({"evaluation_suite_id": result["evaluation_suite_id"], "cases": len(result["cases"]), "output": str(Path(args.output).resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
