"""Vistas de depuración discretizadas desde el estado físico continuo."""

import numpy as np


def _grid_dimensions(env):
    return tuple(int(value) for value in env.config["world"]["grid_cells"])


def _cell_for(position, cell_size, dimensions):
    raw = np.floor(np.asarray(position) / cell_size).astype(int)
    return tuple(np.clip(raw, 0, np.asarray(dimensions) - 1))


def top_view(env, path):
    world = env.config["world"]
    dimensions = _grid_dimensions(env)
    nx, ny, _ = dimensions
    cell_size = float(world["cell_size_m"])
    obstacle_cells = set()
    for minimum, maximum in env._obstacles:
        x0, y0, _ = _cell_for(minimum, cell_size, dimensions)
        x1, y1, _ = _cell_for(maximum, cell_size, dimensions)
        obstacle_cells.update((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))
    path_cells = {_cell_for(point, cell_size, dimensions)[:2] for point in path}
    base = _cell_for(world["base_center_m"], cell_size, dimensions)[:2]
    delivery = _cell_for(world["delivery_center_m"], cell_size, dimensions)[:2]
    rows = []
    for y in range(ny - 1, -1, -1):
        row = ""
        for x in range(nx):
            cell = (x, y)
            if cell == base:
                row += " BAS "
            elif cell == delivery:
                row += " DEL "
            elif cell in obstacle_cells:
                row += " ### "
            elif cell in path_cells:
                row += "  ·  "
            else:
                row += "  .  "
        rows.append(row)
    return "\n".join(rows)


def side_view(env, path):
    world = env.config["world"]
    dimensions = _grid_dimensions(env)
    nx, _, nz = dimensions
    cell_size = float(world["cell_size_m"])
    obstacle_cells = set()
    for minimum, maximum in env._obstacles:
        x0, _, z0 = _cell_for(minimum, cell_size, dimensions)
        x1, _, z1 = _cell_for(maximum, cell_size, dimensions)
        obstacle_cells.update((x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1))
    path_cells = {_cell_for(point, cell_size, dimensions)[::2] for point in path}
    base = _cell_for(world["base_center_m"], cell_size, dimensions)[::2]
    delivery = _cell_for(world["delivery_center_m"], cell_size, dimensions)[::2]
    rows = []
    for z in range(nz - 1, -1, -1):
        row = f"{z * cell_size:5.1f}m "
        for x in range(nx):
            cell = (x, z)
            if cell == base:
                row += " BAS "
            elif cell == delivery:
                row += " DEL "
            elif cell in obstacle_cells:
                row += " ### "
            elif cell in path_cells:
                row += "  ·  "
            else:
                row += "  .  "
        rows.append(row)
    return "\n".join(rows)


def show(env, path):
    print("\nVista superior: BAS=base, DEL=entrega\n" + top_view(env, path))
    print("\nPerfil x-altura\n" + side_view(env, path))
