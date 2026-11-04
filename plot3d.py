"""Vista 3D de una ruta física. Uso: python plot3d.py [--show]"""

import argparse
import sys
from pathlib import Path
from uuid import uuid4

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import envs  # noqa: F401
from train import greedy_path, train


def draw(path, env, out=None, show_window=False):
    world = env.config["world"]
    world_size = np.asarray(world["grid_cells"], dtype=np.float64) * float(world["cell_size_m"])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for minimum, maximum in env._obstacles:
        x0, y0, z0 = minimum
        x1, y1, z1 = maximum
        for z in (z0, z1):
            x_grid, y_grid = np.meshgrid([x0, x1], [y0, y1])
            ax.plot_surface(x_grid, y_grid, np.full_like(x_grid, z), color="tab:gray", alpha=0.35)
        for x in (x0, x1):
            y_grid, z_grid = np.meshgrid([y0, y1], [z0, z1])
            ax.plot_surface(np.full_like(y_grid, x), y_grid, z_grid, color="tab:gray", alpha=0.25)
        for y in (y0, y1):
            x_grid, z_grid = np.meshgrid([x0, x1], [z0, z1])
            ax.plot_surface(x_grid, np.full_like(x_grid, y), z_grid, color="tab:gray", alpha=0.25)

    points = np.asarray(path, dtype=np.float64)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="tab:blue", lw=2.5, label="ruta del dron")
    base = np.asarray(world["base_center_m"], dtype=np.float64)
    delivery = np.asarray(world["delivery_center_m"], dtype=np.float64)
    ax.scatter(*base, s=180, c="limegreen", edgecolors="k", label="BASE")
    ax.scatter(*delivery, s=180, c="red", edgecolors="k", label="zona de entrega")
    ax.set(xlim=(0, world_size[0]), ylim=(0, world_size[1]), zlim=(0, world_size[2]),
           xlabel="x (m)", ylabel="y (m)", zlabel="altura (m)")
    ax.set_box_aspect((world_size[0], world_size[1], world_size[2] * 0.6))
    ax.view_init(elev=28, azim=-55)
    ax.legend(loc="upper left")
    ax.set_title(f"Drone Rescue: ruta ({len(path) - 1} pasos)")
    if out is None:
        out = Path("outputs") / f"drone_route_{uuid4().hex[:8]}.png"
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"La salida ya existe; elige una ruta nueva: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("guardado en", out)
    if show_window:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza una ruta Drone Rescue")
    parser.add_argument("--show", action="store_true", help="abre la ventana de Matplotlib")
    parser.add_argument("--out", type=Path, help="ruta nueva para guardar la imagen")
    args = parser.parse_args()
    environment, agent, _ = train()
    path, result = greedy_path(environment, agent)
    print("resultado:", result)
    draw(path, environment.unwrapped, out=args.out, show_window=args.show)
