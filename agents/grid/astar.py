"""Línea base: ruta de menor coste con A* (mismas reglas de choque y mismo coste que el entorno).

En grid-drone-v1 el coste es el paso más la penalización por volar bajo. En grid-drone-v2
(DroneStormEnv) el entorno define plan_cost (energía del movimiento con el viento medio) y
plan_heuristic; las ráfagas no se pueden planear, por eso AStarReplanner vuelve a planear en
cada paso desde donde quedó el dron.
"""
import heapq

from envs.grid_drone_env import ACTIONS, GOAL, R_LOW, R_STEP, START, dist_to_goal


def move_cost(env, q):
    """Coste de entrar en la celda q: el paso más la penalización por volar bajo (si aplica)."""
    low = q[2] < env.min_cruise_z and (q[0], q[1]) not in (START[:2], GOAL[:2])
    return -R_STEP - (R_LOW if low else 0.0)


def astar(env, start=None, risk_weight=0.0):
    """Lista de celdas de menor coste desde start (por defecto, A) hasta B, o None si no hay ruta.

    En DroneStormEnv usa su plan_route (Dijkstra vectorizado, mismo coste; ver tests), que además
    admite risk_weight para penalizar las celdas desde donde una ráfaga empujaría contra un edificio.
    """
    if hasattr(env, "plan_route"):
        return env.plan_route(start, risk_weight)
    start = tuple(start) if start is not None else getattr(env, "start", START)
    goal = getattr(env, "goal", GOAL)
    cost = getattr(env, "plan_cost", None) or (lambda p, a, q: move_cost(env, q))
    heuristic = getattr(env, "plan_heuristic", None) or (lambda q: dist_to_goal(q) * -R_STEP)
    open_set = [(heuristic(start), 0.0, start)]
    came, best = {start: None}, {start: 0.0}
    while open_set:
        _, g, p = heapq.heappop(open_set)
        if p == goal:
            path = []
            while p is not None:
                path.append(p)
                p = came[p]
            return path[::-1]
        if g > best[p]:
            continue
        for action, (dx, dy, dz) in enumerate(ACTIONS):
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            if env.blocked(*q):
                continue
            g2 = g + cost(p, action, q)
            if g2 < best.get(q, 1e9):
                best[q], came[q] = g2, p
                heapq.heappush(open_set, (g2 + heuristic(q), g2, q))
    return None


class AStarReplanner:
    """Política de referencia que conoce el mapa completo: sigue A* y replanea tras cada paso.

    Con risk_weight=0 no modela el riesgo de las ráfagas (planea con el viento medio) y puede pasar
    pegado a un edificio contra el que el viento lo estrelle; con risk_weight > 0 lo evita a cambio
    de gastar más batería. Es una referencia con información total del mapa, no un agente.
    """

    def __init__(self, env, risk_weight=0.0):
        self.env = env.unwrapped
        self.risk_weight = risk_weight

    def act(self, obs=None):
        del obs
        route = astar(self.env, start=self.env.pos, risk_weight=self.risk_weight)
        if not route or len(route) < 2:
            return 4  # sin ruta: subir es lo menos malo
        step = tuple(b - a for a, b in zip(route[0], route[1]))
        return ACTIONS.index(step)
