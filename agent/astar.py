"""Línea base: ruta legal más corta de A a B con A* (mismas reglas de choque que el entorno)."""
import heapq

from env.drone_env import ACTIONS, GOAL, START, dist_to_goal


def astar(env):
    """Devuelve la lista de celdas de A a B, o None si no hay ruta."""
    open_set = [(dist_to_goal(START), 0, START)]
    came, cost = {START: None}, {START: 0}
    while open_set:
        _, g, p = heapq.heappop(open_set)
        if p == GOAL:
            path = []
            while p is not None:
                path.append(p)
                p = came[p]
            return path[::-1]
        if g > cost[p]:
            continue
        for dx, dy, dz in ACTIONS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            if env.blocked(*q) or cost.get(q, 1e9) <= g + 1:
                continue
            cost[q], came[q] = g + 1, p
            heapq.heappush(open_set, (g + 1 + dist_to_goal(q), g + 1, q))
    return None
