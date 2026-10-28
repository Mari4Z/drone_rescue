"""Genera un HTML autónomo (three.js) con el entorno y la ruta embebidos."""
import json
from pathlib import Path

from env.drone_env import CELL, GOAL, NX, NY, NZ, START

TEMPLATE = Path(__file__).with_name("template.html")


def export_html(env, path, result, env_id, out="drone_view.html"):
    data = {
        "env": env_id,
        "cell": CELL,
        "size": [NX, NY, NZ],
        "start": list(START),
        "goal": list(GOAL),
        "columns": [list(c) for c in env.columns],
        "path": [list(p) for p in path],
        "result": result,
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data))
    Path(out).write_text(html, encoding="utf-8")
    return out
