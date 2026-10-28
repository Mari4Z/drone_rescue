"""Línea base: ruta óptima de A a B con A* (mismas reglas de choque y mismo coste que el entorno)."""
import heapq

from env.drone_env import ACTIONS, GOAL, R_LOW, R_STEP, START, dist_to_goal


def move_cost(env, q):
    """Coste de entrar en la celda q: el paso más la penalización por volar bajo (si aplica)."""
    low = q[2] < env.min_cruise_z and (q[0], q[1]) not in (START[:2], GOAL[:2])
    return -R_STEP - (R_LOW if low else 0.0)


def astar(env):
    """Devuelve la lista de celdas de A a B de menor coste, o None si no hay ruta."""
    open_set = [(dist_to_goal(START), 0.0, START)]
    came, cost = {START: None}, {START: 0.0}
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
            if env.blocked(*q):
                continue
            g2 = g + move_cost(env, q)
            if g2 < cost.get(q, 1e9):
                cost[q], came[q] = g2, p
                heapq.heappush(open_set, (g2 + dist_to_goal(q) * -R_STEP, g2, q))
    return None
