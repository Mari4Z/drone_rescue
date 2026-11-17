"""Sonda de dificultad de DroneRescueGridCityHard-v0 (grid-drone-v3, ambiente 3.1.0).

Mide con A* cuánto estorba cada configuración de postes, cables y aves, para decidir la dificultad
antes de entrenar. Es lo que justifica los valores por omisión de DroneCityHardEnv
(docs/familia_cuadricula.md). Usa la misma clase que el entrenamiento, con otros parámetros.

- Postes: fracción de mapas cuya ruta óptima cambia al añadirlos, energía extra y cuántos cruces pasan
  por debajo o por encima del cable.
- Aves: éxito de un A* ciego (no ve las aves) y de uno consciente (conoce posición y rumbo de todas).
  El ciego mide cuánto pesan las aves; el consciente, cuánto se puede esquivar con información total
  (no es una cota: sus rodeos gastan batería).

    python scripts/probe_grid_difficulty.py            # ~6 min con 8 procesos
Escribe artifacts/results/grid_city_dificultad_sonda.json.
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import envs.grid_drone_env as g  # noqa: E402
from agents.grid.astar import AStarReplanner, AwareAStarReplanner  # noqa: E402

FIRST_SEED = 700000
CORNER = dict(start=(0, 0, 0), battery_margin=g.BATTERY_MARGIN)  # lo que tenía el ambiente 3.0.0
NO_POLES = dict(pole_top=2, cable_z=(1, 1), long_lines=False, n_lines=g.CITY_POLE_LINES)  # postes de 10 m, como en 3.0.0

POLES = {
    "P0 ambiente 3.0.0: poste 10 m, cable 5-10 m, tendidos cortos": dict(**CORNER, pole_top=2, cable_z=(1, 1), long_lines=False, n_lines=g.CITY_POLE_LINES),
    "P1 poste 30 m, cable 20-30 m, tendidos cortos": dict(**CORNER, pole_top=6, cable_z=(4, 5), long_lines=False, n_lines=g.CITY_POLE_LINES),
    "P2 P1 + tendidos largos (4 a 7, cruzan el mapa)": dict(**CORNER, pole_top=6, cable_z=(4, 5), long_lines=True),
    "P3 poste 40 m, cable 15-35 m, tendidos largos": dict(**CORNER, pole_top=8, cable_z=(3, 6), long_lines=True),
}
BIRDS = {
    "B0 ambiente 3.0.0: 0-6 aves al azar, misma celda": dict(**CORNER, **NO_POLES, bird_mode="walk", bird_radius=0, n_birds=g.CITY_BIRDS),
    "B0b 6 aves al azar, misma celda": dict(**CORNER, **NO_POLES, bird_mode="walk", bird_radius=0, n_birds=(6, 6)),
    "B1 4-6 aves cerca de la ruta, rumbo persistente": dict(**CORNER, **NO_POLES, bird_mode="persist", bird_radius=0),
    "B2 B1 + zona de peligro de 1 celda (5 m)": dict(**CORNER, **NO_POLES, bird_mode="persist", bird_radius=1),
    "B3 B2 + 2 celdas por paso": dict(**CORNER, **NO_POLES, bird_mode="persist", bird_radius=1, bird_speed=2),
}
COMBINED = {
    "C1 P2 + B2 (A en la esquina, margen de batería anterior)": dict(**CORNER, pole_top=6, cable_z=(4, 5), long_lines=True, bird_mode="persist", bird_radius=1),
    "H ambiente 3.1.0 (valores por omisión: A en (4, 4), margen 1.3-1.7)": dict(),
}


def poles_job(spec):
    name, kwargs, n = spec
    env = g.DroneCityHardEnv(n_birds=(0, 0), **kwargs)
    changed, extra, lines, crossings, under, over = 0, [], [], 0, 0, 0
    for seed in range(FIRST_SEED, FIRST_SEED + n):
        env.reset(seed=seed)
        with_poles = env.plan_route()
        keep = env.static_obstacles
        env.static_obstacles = lambda: env.obstacles  # el mismo mapa sin postes ni cables
        without = env.plan_route()
        env.static_obstacles = keep
        changed += with_poles != without
        extra.append((env.route_energy(with_poles) - env.route_energy(without)) / env.route_energy(without))
        lines.append(len(env.lines))
        cables, inside = set(env.cables), False
        for cell in with_poles:
            now = (cell[0], cell[1]) in cables
            if now and not inside:
                crossings += 1
                under += cell[2] < env.cable_z[0]
                over += cell[2] > env.cable_z[1]
            inside = now
    return {"kind": "poles", "name": name, "maps": n, "route_changes": changed / n, "energy_extra": float(np.mean(extra)),
            "lines": float(np.mean(lines)), "crossings_per_map": crossings / n, "under": int(under), "over": int(over)}


def birds_job(spec):
    name, kwargs, n = spec
    out = {"kind": "birds", "name": name, "maps": n}
    for label, aware in (("ciego", False), ("consciente", True)):
        env = g.DroneCityHardEnv(**kwargs)
        causes = {"exito": 0, "ave": 0, "viento": 0, "otro": 0}
        for seed in range(FIRST_SEED, FIRST_SEED + n):
            env.reset(seed=seed)
            policy = AwareAStarReplanner(env, risk_weight=30) if aware else AStarReplanner(env, risk_weight=30)
            done = False
            while not done:
                _, _, terminated, truncated, info = env.step(policy.act(None))
                done = terminated or truncated
            key = "exito" if info["result"] == "goal" else {"ave": "ave", "viento": "viento"}.get(info.get("cause"), "otro")
            causes[key] += 1
        out[label] = {k: v / n for k, v in causes.items()}
    return out


if __name__ == "__main__":
    jobs = [(poles_job, (k, v, 150)) for k, v in POLES.items()] + [(birds_job, (k, v, 100)) for k, v in {**BIRDS, **COMBINED}.items()]
    with Pool(8) as pool:
        pending = [pool.apply_async(f, (spec,)) for f, spec in jobs]
        results = [r.get() for r in pending]
    print("== POSTES Y CABLES (150 mapas, sin aves)")
    for r in (x for x in results if x["kind"] == "poles"):
        print(f"{r['name']}\n    la ruta óptima cambia en {r['route_changes']:.0%} de los mapas | energía extra {r['energy_extra']:+.1%} | "
              f"tendidos {r['lines']:.1f} | cruces/mapa {r['crossings_per_map']:.1f} (por debajo {r['under']}, por encima {r['over']})")
    for title, names in (("AVES (100 mapas)", BIRDS), ("COMBINADO (100 mapas)", COMBINED)):
        print(f"== {title}: A* ciego (no ve aves) frente a A* consciente (las ve todas)")
        for r in (x for x in results if x["kind"] == "birds" and x["name"] in names):
            b, a = r["ciego"], r["consciente"]
            print(f"{r['name']}\n    ciego: éxito {b['exito']:.0%}, aves {b['ave']:.0%}, viento {b['viento']:.0%}, otro {b['otro']:.0%} | "
                  f"consciente: éxito {a['exito']:.0%}, aves {a['ave']:.0%}, viento {a['viento']:.0%}, otro {a['otro']:.0%}")
    out = ROOT / "artifacts" / "results" / "grid_city_dificultad_sonda.json"
    out.write_text(json.dumps({"first_seed": FIRST_SEED, "environment_version": g.CITY_HARD_ENVIRONMENT_VERSION, "results": results},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("escrito", out.relative_to(ROOT))
