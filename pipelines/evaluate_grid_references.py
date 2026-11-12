"""Evalúa las referencias de la suite grid-storm-public y deja el resultado versionado.

- astar: conoce el mapa completo y replanea en cada paso, pero no modela las ráfagas.
- astar_viento: además evita las celdas con un edificio a sotavento (risk_weight 30).
- dqn_memoria_v1: el DQN de grid-drone-v1 (run dd7b29c6), que recibe solo las 46 primeras
  entradas porque no conoce el viento ni la batería. Mide cuánto se degrada en el ambiente nuevo.

    python -m pipelines.evaluate_grid_references
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agents.grid.astar import AStarReplanner
from agents.grid.dqn import DQNAgent
from agents.grid.evaluate import evaluate_storm
from pipelines.train import _json_bytes
from pipelines.train_ppo import PROJECT_ROOT, _sha256

SUITE = PROJECT_ROOT / "configs" / "grid_storm_evaluation_suite.yaml"
OUTPUT = PROJECT_ROOT / "artifacts" / "results" / "grid_storm_referencias.json"
V1_RUN = "dd7b29c6bf444d74b072d19e1598e7ab"


def main() -> None:
    suite = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    env_id, first, episodes = suite["environment"], int(suite["first_seed"]), int(suite["episodes"])
    manifest = json.loads((PROJECT_ROOT / "manifests" / f"{V1_RUN}.json").read_text(encoding="utf-8"))
    weights = PROJECT_ROOT / manifest["package_path"] / Path(manifest["model_uri"]).name
    if _sha256(weights) != manifest["model_sha256"]:
        raise ValueError("El DQN de grid-drone-v1 no coincide con su manifiesto")
    old = DQNAgent.load(weights, "cpu")
    cases = []
    for policy_id, kwargs, note in (
        ("astar", {"make_policy": lambda e: AStarReplanner(e)}, "mapa completo, replanea cada paso, no modela ráfagas"),
        ("astar_viento", {"make_policy": lambda e: AStarReplanner(e, risk_weight=30)}, "además evita edificios a sotavento"),
        ("dqn_memoria_v1", {"policy": lambda obs: old.act(obs[:46])}, f"run {V1_RUN}, solo las 46 entradas de v1"),
    ):
        result = evaluate_storm(kwargs.get("policy"), env_id, episodes=episodes, seed=first, make_policy=kwargs.get("make_policy"))
        cases.append({"case_id": f"{env_id}-{policy_id}", "policy_id": policy_id, "note": note, **result})
        m = result["metrics"]
        print(policy_id, f"éxito {m['success_rate']:.1%}", m["cause_counts"], flush=True)
    output = {"evaluation_suite_id": suite["evaluation_suite_id"], "evaluation_suite_version": suite["version"],
              "evaluation_suite_sha256": _sha256(SUITE), "environment": env_id, "cases": cases}
    OUTPUT.write_bytes(_json_bytes(output))
    print("escrito", OUTPUT.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
