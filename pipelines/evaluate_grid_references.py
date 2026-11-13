"""Evalúa las referencias de las suites de cuadrícula y deja el resultado versionado.

grid-storm-public (grid-drone-v2):
- astar: conoce el mapa completo y replanea en cada paso, pero no modela las ráfagas.
- astar_viento: además evita las celdas con un edificio a sotavento (risk_weight 30).
- dqn_memoria_v1: el DQN de grid-drone-v1 (run dd7b29c6), solo con sus 46 entradas.

grid-city-public (grid-drone-v3):
- astar_viento: igual, con postes y cables en el mapa; no ve las aves.
- dqn_storm_v2: el mejor DQN de grid-drone-v2 (run 3b95307e), con sus 51 entradas, sin ver aves.

    python -m pipelines.evaluate_grid_references            # las dos suites
    python -m pipelines.evaluate_grid_references city       # solo una
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from agents.grid.astar import AStarReplanner
from agents.grid.dqn import DQNAgent
from agents.grid.evaluate import evaluate_storm
from pipelines.train import _json_bytes
from pipelines.train_ppo import PROJECT_ROOT, _sha256

SUITES = {
    "storm": (PROJECT_ROOT / "configs" / "grid_storm_evaluation_suite.yaml", PROJECT_ROOT / "artifacts" / "results" / "grid_storm_referencias.json"),
    "city": (PROJECT_ROOT / "configs" / "grid_city_evaluation_suite.yaml", PROJECT_ROOT / "artifacts" / "results" / "grid_city_referencias.json"),
}
V1_RUN = "dd7b29c6bf444d74b072d19e1598e7ab"
V2_RUN = "3b95307e890245d3b011bfda923b9977"


def _load_dqn(run_id: str) -> DQNAgent:
    manifest = json.loads((PROJECT_ROOT / "manifests" / f"{run_id}.json").read_text(encoding="utf-8"))
    weights = PROJECT_ROOT / manifest["package_path"] / Path(manifest["model_uri"]).name
    if _sha256(weights) != manifest["model_sha256"]:
        raise ValueError(f"El modelo {run_id} no coincide con su manifiesto")
    return DQNAgent.load(weights, "cpu")


def references(name: str) -> list[tuple[str, dict, str]]:
    if name == "storm":
        old = _load_dqn(V1_RUN)
        return [
            ("astar", {"make_policy": lambda e: AStarReplanner(e)}, "mapa completo, replanea cada paso, no modela ráfagas"),
            ("astar_viento", {"make_policy": lambda e: AStarReplanner(e, risk_weight=30)}, "además evita edificios a sotavento"),
            ("dqn_memoria_v1", {"policy": lambda obs: old.act(obs[:46])}, f"run {V1_RUN}, solo las 46 entradas de v1"),
        ]
    storm = _load_dqn(V2_RUN)
    return [
        ("astar_viento", {"make_policy": lambda e: AStarReplanner(e, risk_weight=30)}, "mapa completo con postes y cables, evita sotavento, no ve aves"),
        ("dqn_storm_v2", {"policy": lambda obs: storm.act(obs[:51])}, f"run {V2_RUN}, solo las 51 entradas de v2, no ve aves"),
    ]


def main(names: list[str]) -> None:
    for name in names:
        suite_path, output = SUITES[name]
        suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        env_id, first, episodes = suite["environment"], int(suite["first_seed"]), int(suite["episodes"])
        cases = []
        for policy_id, kwargs, note in references(name):
            result = evaluate_storm(kwargs.get("policy"), env_id, episodes=episodes, seed=first, make_policy=kwargs.get("make_policy"))
            cases.append({"case_id": f"{env_id}-{policy_id}", "policy_id": policy_id, "note": note, **result})
            m = result["metrics"]
            print(name, policy_id, f"éxito {m['success_rate']:.1%}", m["cause_counts"], flush=True)
        output.write_bytes(_json_bytes({"evaluation_suite_id": suite["evaluation_suite_id"], "evaluation_suite_version": suite["version"],
                                        "evaluation_suite_sha256": _sha256(suite_path), "environment": env_id, "cases": cases}))
        print("escrito", output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main(sys.argv[1:] or list(SUITES))
