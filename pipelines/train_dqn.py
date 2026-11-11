"""Entrena el DQN sobre un perfil YAML sin cambiar el ambiente ni la recompensa.

    python -m pipelines.train_dqn --profile configs/training/dqn_n1.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.dqn import DQNAgent, DQNPolicy, NStepCollector, ReplayBuffer, make_features
from envs.fast_drone_env import FastDroneEnv


def run_episode(env, policy, seed: int) -> dict:
    observation, _ = env.reset(seed=seed)
    total, done, info = 0.0, False, {}
    while not done:
        observation, reward, terminated, truncated, info = env.step(policy.act(observation))
        total += float(reward)
        done = terminated or truncated
    return {"cause": info["cause"], "stage": info["stage"], "steps": env.steps, "return": total}


def evaluate_greedy(env, agent: DQNAgent, seeds: list[int]) -> dict:
    agent.sync_actor()
    episodes = [run_episode(env, DQNPolicy(agent), seed) for seed in seeds]
    return {
        "success": float(np.mean([e["cause"] == "success" for e in episodes])),
        "delivered": float(np.mean([e["stage"] in ("return", "complete") for e in episodes])),
        "causes": dict(Counter(e["cause"] for e in episodes)),
        "steps": float(np.mean([e["steps"] for e in episodes])),
        "return": float(np.mean([e["return"] for e in episodes])),
    }


def score(metrics: dict) -> tuple:
    """Elige el mejor modelo por éxito y entrega; a igualdad, retorno. Nunca solo por retorno."""
    return metrics["success"], metrics["delivered"], metrics["return"]


def train(profile: dict, seed: int, device: str, out: Path, log=print) -> dict:
    t = profile["training"]
    env = FastDroneEnv(config_path=PROJECT_ROOT / profile["level"]["config"])
    eval_env = FastDroneEnv(config_path=PROJECT_ROOT / profile["level"]["config"])
    agent = DQNAgent(device=device, lr=float(t["lr"]), hidden=int(t["hidden"]), seed=seed)
    buffer = ReplayBuffer(int(t["buffer_size"]), agent.device)
    collector = NStepCollector(buffer, int(t["n_step"]), float(t["gamma"]))
    rng = np.random.default_rng(seed)
    steps = int(t["steps"])
    eps_end, explore_until = float(t["eps_end"]), float(t["eps_fraction"]) * steps
    reward_scale, batch_size = float(t["reward_scale"]), int(t["batch_size"])

    observation, _ = env.reset(seed=seed)
    features = make_features(observation)
    recent, losses = deque(maxlen=200), deque(maxlen=1000)
    best, best_metrics, started = None, None, time.time()
    hold_action, hold_left = 0, 0

    for step in range(1, steps + 1):
        eps = max(eps_end, 1.0 - (1.0 - eps_end) * step / explore_until)
        if hold_left > 0:
            action, hold_left = hold_action, hold_left - 1
        elif rng.random() < eps:
            # Exploración sostenida: con inercia, una acción suelta apenas mueve al dron
            action = hold_action = int(rng.integers(7))
            hold_left = int(rng.geometric(1.0 / float(t["explore_hold"]))) - 1
        else:
            action = agent.greedy(features)

        observation, reward, terminated, truncated, info = env.step(action)
        next_features = make_features(observation)
        collector.add(features, action, float(reward) * reward_scale, next_features, terminated, truncated)
        features = next_features
        if terminated or truncated:
            recent.append((info["cause"], info["stage"]))
            observation, _ = env.reset()
            features = make_features(observation)
            hold_left = 0

        if step >= int(t["learning_starts"]) and step % int(t["train_freq"]) == 0:
            losses.append(agent.learn(buffer.sample(batch_size)))
        if step % int(t["target_every"]) == 0:
            agent.sync_target()
        if step % int(t["actor_every"]) == 0:
            agent.sync_actor()

        if step % int(t["eval_every"]) == 0:
            metrics = evaluate_greedy(eval_env, agent, [int(s) for s in t["evaluation_seeds"]])
            causes = Counter(c for c, _ in recent)
            n = max(1, len(recent))
            delivered = sum(s in ("return", "complete") for _, s in recent) / n
            loss = torch.stack(list(losses)).mean().item() if losses else 0.0
            line = (f"paso {step:8d} | eps {eps:.2f} | train éxito {causes['success'] / n:5.1%} entrega {delivered:5.1%}"
                    f" | loss {loss:.3f} | eval éxito {metrics['success']:4.0%} entrega {metrics['delivered']:4.0%}"
                    f" pasos {metrics['steps']:4.0f} ret {metrics['return']:7.1f} {metrics['causes']}"
                    f" | {time.time() - started:.0f}s")
            if best is None or score(metrics) > best:
                best, best_metrics = score(metrics), metrics
                agent.save(out)
                line += " *guardado*"
            log(line)
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto (GPU si hay), cpu, mps o cuda")
    parser.add_argument("--steps", type=int, help="sustituye training.steps del perfil")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    profile = yaml.safe_load(Path(args.profile).read_text(encoding="utf-8"))
    if args.steps:
        profile["training"]["steps"] = args.steps
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"DQN en {profile['level']['id']} | semilla {args.seed} | {profile['training']['steps']} pasos", flush=True)
    best = train(profile, args.seed, args.device, out, log=lambda s: print(s, flush=True))
    print("mejor evaluación:", best, "| modelo:", out, flush=True)


if __name__ == "__main__":
    main()
