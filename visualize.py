"""Visualización en texto: vista superior y perfil de la ruta, con A y B resaltados."""
import numpy as np

from env.drone_env import GOAL, NX, NY, NZ, START

A_MARK, B_MARK = "A", "B"


def top_view(env, path):
    """Vista desde arriba: # columna, · ruta, A = inicio, B = meta."""
    rows = []
    cols = env.obstacles.any(axis=2)
    on_path = {(p[0], p[1]) for p in path}
    for y in range(NY - 1, -1, -1):
        row = ""
        for x in range(NX):
            if (x, y) == (START[0], START[1]):
                row += f"[{A_MARK}]"
            elif (x, y) == (GOAL[0], GOAL[1]):
                row += f"[{B_MARK}]"
            elif cols[x, y]:
                row += "###"
            elif (x, y) in on_path:
                row += " · "
            else:
                row += " . "
        rows.append(row)
    return "\n".join(rows)


def side_view(env, path):
    """Perfil (x vs altura): # columna (máx. en y), · ruta."""
    prof = env.obstacles.any(axis=1)
    on_path = {(p[0], p[2]) for p in path}
    rows = []
    for z in range(NZ - 1, -1, -1):
        row = f"{z * 5:3d}m "
        for x in range(NX):
            if (x, z) == (START[0], START[2]):
                row += f"[{A_MARK}]"
            elif (x, z) == (GOAL[0], GOAL[2]):
                row += f"[{B_MARK}]"
            elif (x, z) in on_path:
                row += " · "
            elif prof[x, z]:
                row += "###"
            else:
                row += " . "
        rows.append(row)
    return "\n".join(rows)


def show(env, path):
    print(f"\nVista superior (A={START}, B={GOAL})\n" + top_view(env, path))
    print("\nPerfil (x vs altura)\n" + side_view(env, path))
