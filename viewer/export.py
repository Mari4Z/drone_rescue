"""Genera un HTML autónomo (three.js) con el entorno y uno o varios intentos embebidos."""
import json
from pathlib import Path

from envs.grid_drone_env import CELL, GOAL, NX, NY, NZ, START

TEMPLATE = Path(__file__).with_name("template.html")


def export_html(episodes, env_id, out="drone_view.html"):
    """episodes: lista de dicts con columns, path, result y opcionalmente seed y astar (pasos de A*)."""
    data = {
        "env": env_id,
        "cell": CELL,
        "size": episodes[0].get("size", [NX, NY, NZ]) if episodes else [NX, NY, NZ],
        "start": list(START),
        "goal": list(GOAL),
        "episodes": [
            {**e, "columns": [list(c) for c in e["columns"]], "path": [list(p) for p in e["path"]]}
            for e in episodes
        ],
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data))
    Path(out).write_text(html, encoding="utf-8")
    return out
