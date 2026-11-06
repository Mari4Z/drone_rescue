"""Imagen 3D (matplotlib) de una ruta de la familia de cuadrícula."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.grid_drone_env import CELL, GOAL, NX, NY, NZ, START


def draw(path, columns, out="drone_route.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for x0, x1, y0, y1, h in columns:
        xs, ys = np.array([x0, x1 + 1]) * CELL, np.array([y0, y1 + 1]) * CELL
        X, Y = np.meshgrid(xs, ys)
        for z in (0, h):
            ax.plot_surface(X, Y, np.full_like(X, z, dtype=float), color="tab:gray", alpha=0.6)
        for x in xs:
            ax.plot_surface(np.full((2, 2), x), Y, np.array([[0, 0], [h, h]]).T, color="tab:gray", alpha=0.5)
        for y in ys:
            ax.plot_surface(X, np.full((2, 2), y), np.array([[0, h], [0, h]]), color="tab:gray", alpha=0.5)

    p = (np.array(path) + 0.5) * CELL
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color="tab:blue", lw=2.5, label="ruta del dron")
    a, b = (np.array(START) + 0.5) * CELL, (np.array(GOAL) + 0.5) * CELL
    ax.scatter(*a, s=250, c="limegreen", edgecolors="k", label="A (inicio)")
    ax.scatter(*b, s=250, c="red", edgecolors="k", label="B (meta)")
    ax.text(*(a + [0, 0, 4]), "A", fontsize=14, weight="bold")
    ax.text(*(b + [0, 0, 4]), "B", fontsize=14, weight="bold")

    ax.set(xlim=(0, NX * CELL), ylim=(0, NY * CELL), zlim=(0, NZ * CELL), xlabel="x (m)", ylabel="y (m)", zlabel="altura (m)")
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=28, azim=-55)
    ax.legend(loc="upper left")
    ax.set_title(f"Drone Rescue: ruta aprendida ({len(path) - 1} pasos)")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("guardado en", out)
