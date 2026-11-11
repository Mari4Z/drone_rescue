"""Genera el dataset del proyecto: escenarios y episodios de la suite de evaluación.

En aprendizaje por refuerzo los datos se generan al interactuar con el ambiente, así que el
dataset versiona lo que la guía llama generador, distribuciones, escenarios y semillas (§10):

- data/escenarios/<nivel>.json: mundo, columnas, base, zona de entrega, batería, distribución del
  viento y, por cada semilla de la suite, el viento sorteado y la observación inicial.
- data/episodios/<politica>.parquet: trayectorias por decisión (observación de 19 valores, acción,
  recompensa, terminación y estado físico) de cada nivel y semilla de la suite.
- data/dataset.json: versiones del generador, commit, suite, políticas y SHA-256 de cada archivo.

Políticas: el modelo oficial (manifests/published), la aleatoria y la regla reactiva de la suite.
Se versiona con DVC (data.dvc):

    python -m pipelines.build_dataset
    dvc add data && git add data.dvc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agents.ppo import make_env
from envs.drone_env import ENVIRONMENT_VERSION, DroneEnv, load_config
from envs.stabilized_drone_env import ACTION_NAMES, STABILIZER_VERSION
from pipelines.evaluate import DEFAULT_SUITE_PATH, RandomPolicy, RulePolicy, load_suite

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data"
DATASET_VERSION = "1.0.0"
FORCE_ACTIONS = ["+x", "-x", "+y", "-y", "subir", "bajar", "soltar"]
OBSERVATION_NAMES = ([f"objetivo_{a}_m" for a in "xyz"] + [f"velocidad_{a}_mps" for a in "xyz"] + ["altura_m"]
                     + [f"sensor_{d}_m" for d in ("+x", "-x", "+y", "-y", "+z", "-z")]
                     + ["bateria_frac", "carga"] + [f"etapa_{s}" for s in ("outbound", "delivery", "return", "complete")])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_model() -> dict:
    """Modelo vigente según la última publicación de N1."""
    publication = json.loads(sorted((ROOT / "manifests" / "published" / "N1").glob("v*.json"))[-1].read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "manifests" / f"{publication['run_id']}.json").read_text(encoding="utf-8"))
    model = ROOT / manifest["package_path"] / "model.zip"
    if sha256(model) != manifest["model_sha256"]:
        raise ValueError("El modelo oficial no coincide con su manifiesto")
    return {"publication": publication, "manifest": manifest, "path": model}


def scenario(level: dict, seeds: list[int]) -> dict:
    config_path = ROOT / level["config"]
    config = load_config(config_path)
    env = DroneEnv(config_path)
    episodes = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        episodes.append({"seed": seed, "viento_mps2": env.wind.round(6).tolist(),
                         "observacion_inicial": dict(zip(OBSERVATION_NAMES, np.round(observation, 6).tolist()))})
    return {"level_id": level["id"], "config": level["config"], "config_sha256": sha256(config_path),
            "environment_version": ENVIRONMENT_VERSION, "reward_version": str(config["reward"]["version"]),
            "mundo": config["world"], "fisica": config["physics"], "viento": config["wind"], "bateria": config["battery"],
            "mision": config["mission"], "sensores": config["sensors"], "episodios_suite": episodes}


def rollout(policy_id: str, level: dict, seed: int, official: dict | None) -> list[dict]:
    """Un episodio de la suite. El modelo oficial usa su familia (v3); las líneas base, la v2."""
    config_path = ROOT / level["config"]
    if policy_id == "oficial":
        from stable_baselines3 import PPO

        model = PPO.load(str(official["path"]), device="cpu")
        env, family, names = make_env(config_path, official["manifest"]["model_family"]), official["manifest"]["model_family"], ACTION_NAMES
    else:
        env, family, names = DroneEnv(config_path), "drone-rescue-v2", FORCE_ACTIONS
        policy = RandomPolicy(len(FORCE_ACTIONS), seed) if policy_id == "aleatoria" else RulePolicy()
        config = load_config(config_path)
    base = env.unwrapped
    observation, _ = env.reset(seed=seed)
    rows, terminated, truncated, step = [], False, False, 0
    while not (terminated or truncated):
        raw = base._observation()  # los 19 valores del ambiente, antes del escalado de la red
        if policy_id == "oficial":
            action = int(model.predict(observation, deterministic=True)[0])
        else:
            action = policy.act(observation, config)
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append({"politica": policy_id, "familia": family, "nivel": level["id"], "semilla": seed, "paso": step, "pasos_fisicos": int(base.steps),
                     **dict(zip(OBSERVATION_NAMES, raw.astype(float))),
                     "accion": action, "accion_nombre": names[action], "recompensa": float(reward),
                     "terminated": bool(terminated), "truncated": bool(truncated), "causa": info.get("cause"),
                     "etapa": base.stage, "pos_x_m": float(base.position[0]), "pos_y_m": float(base.position[1]),
                     "pos_z_m": float(base.position[2]), "bateria_wh": float(base.battery_wh), "contactos": int(base.contacts)})
        step += 1
    env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))
    args = parser.parse_args()
    suite, suite_path = load_suite(args.suite)
    official = official_model()
    (OUTPUT / "escenarios").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "episodios").mkdir(parents=True, exist_ok=True)

    for level in suite["levels"]:
        path = OUTPUT / "escenarios" / f"{level['id']}.json"
        path.write_text(json.dumps(scenario(level, suite["seeds"]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {}
    for policy_id in ("oficial", "aleatoria", "regla"):
        rows = [row for level in suite["levels"] for seed in suite["seeds"] for row in rollout(policy_id, level, seed, official)]
        frame = pd.DataFrame(rows)
        path = OUTPUT / "episodios" / f"{policy_id}.parquet"
        frame.to_parquet(path, index=False)
        last = frame.groupby(["nivel", "semilla"]).tail(1)
        summary[policy_id] = {"pasos": len(frame), "episodios": len(last),
                              "exito_por_nivel": {lv: float((g["causa"] == "success").mean()) for lv, g in last.groupby("nivel")}}

    files = sorted(p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "dataset.json")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    metadata = {
        "dataset_version": DATASET_VERSION, "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "environment_version": ENVIRONMENT_VERSION, "stabilizer_version": STABILIZER_VERSION,
        "suite": {"id": suite["evaluation_suite_id"], "version": suite["version"], "sha256": sha256(suite_path), "seeds": suite["seeds"]},
        "politicas": {
            "oficial": {"run_id": official["manifest"]["run_id"], "model_sha256": official["manifest"]["model_sha256"],
                        "familia": official["manifest"]["model_family"],
                        "publicacion": f"{official['publication']['level']} v{official['publication']['publication_version']:04d}"},
            "aleatoria": {"familia": "drone-rescue-v2", "detalle": "acción uniforme con la semilla del episodio"},
            "regla": {"familia": "drone-rescue-v2", "detalle": "control reactivo de pipelines/evaluate.py"},
        },
        "columnas_observacion": OBSERVATION_NAMES,
        "resumen": summary,
        "archivos": {str(p.relative_to(OUTPUT)): sha256(p) for p in files},
    }
    (OUTPUT / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resumen": summary, "archivos": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
