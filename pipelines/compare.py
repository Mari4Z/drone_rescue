"""Criba contrastes, reentrena finalistas y separa fuentes de variabilidad."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from envs.drone_env import load_config
from pipelines.evaluate import load_suite
from pipelines.train import PROJECT_ROOT, train_run

DEFAULT_CANDIDATES = ("exp-001", "exp-002", "exp-003", "exp-004")
DEFAULT_SEEDS = (11, 23, 37)
DEFAULT_SCREEN_SEED = 7001
DEFAULT_SCREENING_EPISODES = 500
DEFAULT_DECISION_PATH = PROJECT_ROOT / "docs" / "decision.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "results"
TRAINING_PROFILE_TEMPLATE = PROJECT_ROOT / "configs" / "training" / "n1.yaml"


def _repo_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_candidate(candidate_id: str) -> dict:
    path = PROJECT_ROOT / "configs" / "experiments" / f"{candidate_id}.yaml"
    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    experiment = config.get("experiment", {})
    if experiment.get("id") != candidate_id:
        raise ValueError(f"El id dentro de {path.name} no coincide con {candidate_id}")
    return {"id": candidate_id, "path": str(path.relative_to(PROJECT_ROOT)), **experiment}


def _write_run_profile(candidate: dict, suite: dict, episodes: int, destination: Path) -> Path:
    template_path = TRAINING_PROFILE_TEMPLATE
    with template_path.open(encoding="utf-8") as source:
        profile = yaml.safe_load(source)
    profile["experiment"] = {
        "id": candidate["id"],
        "hypothesis": candidate["hypothesis"],
    }
    profile["level"] = {"id": candidate["id"], "config": candidate["path"]}
    profile["training"]["episodes"] = int(episodes)
    profile["training"]["evaluation_seeds"] = [int(seed) for seed in suite["seeds"]]
    profile_path = destination / f"{candidate['id']}.yaml"
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return profile_path


def _q_cases(evaluation: dict) -> dict[str, dict]:
    cases = {
        case["level_id"]: case
        for case in evaluation["cases"]
        if case["policy_id"] == "q_learning_greedy"
    }
    return cases


def _episode_metric(episode: dict, metric: str) -> float:
    if metric == "success_rate":
        return float(episode["success"])
    if metric.startswith("failure:"):
        return float(episode["cause"] == metric.split(":", 1)[1])
    return float(episode[metric])


def _training_seed_summary(runs: list[dict], metric: str) -> dict:
    per_seed = []
    episode_std = []
    for run in runs:
        episodes = [episode for case in run["evaluation_cases"].values() for episode in case["episodes"]]
        values = np.asarray([_episode_metric(episode, metric) for episode in episodes], dtype=np.float64)
        per_seed.append(float(values.mean()))
        episode_std.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
    return {
        "between_training_seeds": {
            "mean": float(np.mean(per_seed)),
            "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
            "n": len(per_seed),
            "per_seed": per_seed,
        },
        "within_evaluation_episodes": {
            "mean_std": float(np.mean(episode_std)),
            "per_training_seed_std": episode_std,
            "episodes_per_model": sum(len(case["episodes"]) for case in runs[0]["evaluation_cases"].values()) if runs else 0,
        },
    }


def summarize_finalist(runs: list[dict]) -> dict:
    all_episodes = [episode for run in runs for case in run["evaluation_cases"].values() for episode in case["episodes"]]
    causes = sorted({episode["cause"] for episode in all_episodes if episode["cause"] not in {None, "success"}})
    metrics = {
        "success_rate": _training_seed_summary(runs, "success_rate"),
        "steps": _training_seed_summary(runs, "steps"),
        "battery_remaining_wh": _training_seed_summary(runs, "battery_remaining_wh"),
        "inference_latency_ms": _training_seed_summary(runs, "inference_latency_ms"),
    }
    for cause in causes:
        metrics[f"failure:{cause}"] = _training_seed_summary(runs, f"failure:{cause}")
    case_summaries = {}
    for case_id in sorted(runs[0]["evaluation_cases"] if runs else []):
        case_runs = [
            {"evaluation_cases": {case_id: run["evaluation_cases"][case_id]}}
            for run in runs
        ]
        case_summaries[case_id] = {
            metric: _training_seed_summary(case_runs, metric)
            for metric in ["success_rate", "steps", "battery_remaining_wh", "inference_latency_ms", *[f"failure:{cause}" for cause in causes]]
        }
    return {
        "training_seeds": [run["training_seed"] for run in runs],
        "training_seed_count": len(runs),
        "evaluation_episode_count_per_model": len(all_episodes) // max(len(runs), 1),
        "metrics": metrics,
        "by_evaluation_case": case_summaries,
        "runs": [
            {
                **{key: run[key] for key in ("training_seed", "run_id", "registry_uri")},
                "manifest_path": _repo_path(run["manifest_path"]),
                "checkpoint_path": _repo_path(run["checkpoint_path"]),
            }
            for run in runs
        ],
    }


def rank_screening_candidates(screening: list[dict]) -> list[dict]:
    """Ordena por éxito y fallos; retorno no participa en selección."""
    def score(candidate: dict):
        metrics = candidate["screening_metrics"]
        causes = metrics["cause_counts"]
        catastrophic = sum(
            causes.get(cause, 0) for cause in ("collision", "out_of_bounds", "battery_depleted")
        ) / max(metrics["episodes"], 1)
        return (
            -float(metrics["success_rate"]),
            catastrophic,
            float(metrics["mean_steps"]),
            -float(metrics["mean_battery_remaining_wh"]),
            float(metrics["mean_inference_latency_ms"]),
            candidate["candidate_id"],
        )

    return sorted(screening, key=score)


def _percent_cell(summary: dict, multiplier: float = 100.0) -> str:
    seed = summary["between_training_seeds"]
    episode = summary["within_evaluation_episodes"]["mean_std"]
    return f"{seed['mean'] * multiplier:.1f} ± {seed['std'] * multiplier:.1f}; ep σ={episode * multiplier:.1f}"


def _value_cell(summary: dict, decimals: int, suffix: str = "") -> str:
    seed = summary["between_training_seeds"]
    episode = summary["within_evaluation_episodes"]["mean_std"]
    return f"{seed['mean']:.{decimals}f} ± {seed['std']:.{decimals}f}{suffix}; ep σ={episode:.{decimals}f}{suffix}"


def decision_markdown(result: dict) -> str:
    ranked = result["screening"]["ranking"]
    finalists = result["finalists"]
    selected_ids = result["screening"]["selected_finalists"]
    selected = [candidate for candidate in ranked if candidate["candidate_id"] in selected_ids]
    score_pairs = []
    for item in selected:
        score_pairs.append((item["candidate_id"], finalists[item["candidate_id"]]["metrics"]["success_rate"]["between_training_seeds"]["mean"]))
    deployable = [(candidate, success) for candidate, success in score_pairs if success > 0.0]
    if deployable:
        chosen_id, _ = max(deployable, key=lambda item: item[1])
        decision = f"SELECCIONAR {chosen_id} para continuar validación; no implica promoción a producción."
        reason = "La versión seleccionada obtuvo al menos un éxito bajo la suite común y el mayor éxito medio entre semillas de entrenamiento de los dos finalistas."
    else:
        chosen_id = None
        decision = "RECHAZAR ambos finalistas para uso operativo; no hay evidencia de éxito que justifique promover o recuperar una versión anterior."
        reason = "Los dos finalistas registraron 0% de éxito medio en la evaluación común; los candidatos quedan conservados para análisis y nuevos experimentos."
    lines = [
        "# Decisión de comparación de finalistas",
        "",
        f"- Comparación: `{result['comparison_id']}`",
        f"- Suite común: `{result['evaluation_suite_id']}` v`{result['evaluation_suite_version']}` ({result['evaluation_episode_count']} episodios por configuración).",
        f"- Presupuesto final: {result['final_training_budget']} episodios por run; semillas independientes emparejadas: {', '.join(map(str, result['training_seeds']))}.",
        f"- Cribado: {result['screening_budget']} episodios por candidato, semilla {result['screening_seed']}; se conservaron los cuatro candidatos.",
        "",
        f"## Decisión: {decision}",
        "",
        reason,
        "El cribado ordena por tasa de éxito, luego fallos catastróficos, pasos, batería y latencia; el retorno no interviene en la selección. La tabla final reporta variabilidad entre semillas de entrenamiento por separado de la dispersión entre episodios de evaluación.",
        "",
        "## Evidencia del cribado",
        "",
        "| Rango | Candidato | Éxito | Colisión | Fuera de límites | Batería agotada | Pasos medios | Run cribado | Estado |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, candidate in enumerate(ranked, 1):
        metrics = candidate["screening_metrics"]
        causes = metrics["cause_counts"]
        status = "finalista" if candidate["candidate_id"] in selected_ids else "rechazado en cribado; run conservado"
        lines.append(
            f"| {rank} | `{candidate['candidate_id']}` | {metrics['success_rate']:.1%} | "
            f"{causes.get('collision', 0)} | {causes.get('out_of_bounds', 0)} | "
            f"{causes.get('battery_depleted', 0)} | {metrics['mean_steps']:.1f} | "
            f"`{candidate['screening_run_id']}` | {status} |"
        )
    lines.extend([
        "",
        "## Resultados finales: entrenamiento vs. evaluación",
        "",
        "Cada celda muestra media ± desviación estándar de las medias por semilla de entrenamiento; `ep σ` es la desviación típica media dentro de cada modelo entre episodios de evaluación. Las tasas de éxito/fallo se expresan en porcentaje. Duración está en pasos, batería en Wh, latencia en ms. Métricas globales agregan los cuatro casos de la misma suite; el JSON conserva además resultados desglosados por caso.",
        "",
        "| Métrica | " + " | ".join(f"{candidate} (entre semillas; ep σ)" for candidate in selected_ids) + " |",
        "|---|" + "|".join("---" for _ in selected_ids) + "|",
    ])
    rows = [
        ("Éxito (%)", "success_rate", "percent"),
        ("Colisión (%)", "failure:collision", "percent"),
        ("Fuera de límites (%)", "failure:out_of_bounds", "percent"),
        ("Batería agotada (%)", "failure:battery_depleted", "percent"),
        ("Límite de pasos (%)", "failure:step_limit", "percent"),
        ("Duración (pasos)", "steps", "value1"),
        ("Batería restante (Wh)", "battery_remaining_wh", "value2"),
        ("Latencia inferencia (ms/acción)", "inference_latency_ms", "value3"),
    ]
    for label, key, kind in rows:
        cells = []
        for candidate_id in selected_ids:
            summary = finalists[candidate_id]["metrics"].get(key)
            if summary is None:
                cells.append("0.0 ± 0.0; ep σ=0.0" if kind == "percent" else "N/D")
            elif kind == "percent":
                cells.append(_percent_cell(summary))
            elif kind == "value1":
                cells.append(_value_cell(summary, 1))
            elif kind == "value2":
                cells.append(_value_cell(summary, 2))
            else:
                cells.append(_value_cell(summary, 3))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Conservación y trazabilidad", ""])
    for candidate in ranked:
        cid = candidate["candidate_id"]
        if cid not in selected_ids:
            lines.append(f"- `{cid}` rechazado en cribado: run `{candidate['screening_run_id']}`, perfil/config `{candidate['config_path']}`; su run, checkpoint y versión registrada se conservan en MLflow.")
    for candidate_id in selected_ids:
        for run in finalists[candidate_id]["runs"]:
            lines.append(f"- `{candidate_id}` semilla {run['training_seed']}: run `{run['run_id']}`, manifiesto `{_repo_path(run['manifest_path'])}`, Registry `{run['registry_uri']}`.")
    lines.extend([
        "",
        f"Los retornos se guardan por caso en el JSON de comparación, pero no se promedian ni se usan para ordenar candidatos. Todas las configuraciones conservan `reward_version={result['reward_version']}` y el contrato de observación/acción de la misma familia de modelos.",
        "",
    ])
    return "\n".join(lines)


def run_comparison(
    *,
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES,
    screening_budget: int = DEFAULT_SCREENING_EPISODES,
    training_seeds: tuple[int, ...] = DEFAULT_SEEDS,
    screen_seed: int = DEFAULT_SCREEN_SEED,
    final_budget: int | None = None,
    output_path: str | Path | None = None,
    decision_path: str | Path = DEFAULT_DECISION_PATH,
) -> dict:
    if len(candidates) < 2 or len(set(candidates)) != len(candidates):
        raise ValueError("Se requieren al menos dos candidatos únicos")
    if len(training_seeds) != 3 or len(set(training_seeds)) != 3:
        raise ValueError("La comparación final requiere exactamente tres semillas independientes")
    if screening_budget <= 0:
        raise ValueError("El presupuesto de cribado debe ser positivo")

    candidate_configs = [_load_candidate(candidate_id) for candidate_id in candidates]
    declared_budgets = {int(candidate["episode_budget"]) for candidate in candidate_configs}
    if len(declared_budgets) != 1:
        raise ValueError("Los candidatos no comparten el mismo presupuesto declarado")
    final_budget = int(final_budget or next(iter(declared_budgets)))
    if final_budget <= 0:
        raise ValueError("El presupuesto final debe ser positivo")

    suite, suite_path = load_suite(PROJECT_ROOT / "configs" / "finalist_evaluation_suite.yaml")
    suite_candidate_ids = {entry["id"] for entry in suite["levels"]}
    if set(candidates) != suite_candidate_ids:
        raise ValueError("La suite común debe incluir exactamente los candidatos del contraste")
    reward_versions = {
        str(load_config(PROJECT_ROOT / candidate["path"])["reward"]["version"])
        for candidate in candidate_configs
    }
    if len(reward_versions) != 1:
        raise ValueError("No se pueden comparar retornos: los candidatos usan reward_version distintos")
    reward_version = next(iter(reward_versions))

    comparison_id = datetime.now(timezone.utc).strftime("finalistas-%Y%m%dT%H%M%SZ")
    output = Path(output_path) if output_path else DEFAULT_OUTPUT_DIR / f"{comparison_id}.json"
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise FileExistsError(f"El archivo de comparación existe: {output}")
    decision = Path(decision_path)
    if not decision.is_absolute():
        decision = PROJECT_ROOT / decision
    if decision.exists():
        raise FileExistsError(f"El documento de decisión existe; elige una ruta nueva: {decision}")

    screening = []
    final_runs: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory(prefix=f"{comparison_id}-profiles-") as profile_tmp:
        profile_dir = Path(profile_tmp)
        profiles = {
            candidate["id"]: _write_run_profile(candidate, suite, final_budget, profile_dir)
            for candidate in candidate_configs
        }
        for candidate in candidate_configs:
            candidate_id = candidate["id"]
            result = train_run(
                profiles[candidate_id],
                int(screen_seed),
                suite_path=suite_path,
                episodes_override=int(screening_budget),
            )
            q_cases = _q_cases(result["evaluation"])
            if set(q_cases) != suite_candidate_ids:
                raise RuntimeError("La evaluación de cribado no cubrió toda la suite común")
            primary = q_cases[candidate_id]["metrics"]
            screening.append({
                "candidate_id": candidate_id,
                "config_path": candidate["path"],
                "hypothesis": candidate["hypothesis"],
                "screening_seed": int(screen_seed),
                "screening_budget": int(screening_budget),
                "screening_run_id": result["run_id"],
                "screening_registry_uri": result["registry_uri"],
                "screening_metrics": primary,
                "common_suite_metrics": {case_id: case["metrics"] for case_id, case in q_cases.items()},
                "screening_manifest": _repo_path(result["manifest_path"]),
            })

        ranked = rank_screening_candidates(screening)
        selected_ids = [candidate["candidate_id"] for candidate in ranked[:2]]
        candidate_by_id = {candidate["id"]: candidate for candidate in candidate_configs}
        for candidate_id in selected_ids:
            profile = profiles[candidate_id]
            runs = []
            for training_seed in training_seeds:
                run_result = train_run(
                    profile,
                    int(training_seed),
                    suite_path=suite_path,
                )
                q_cases = _q_cases(run_result["evaluation"])
                if set(q_cases) != suite_candidate_ids:
                    raise RuntimeError("La evaluación final no cubrió toda la suite común")
                runs.append({
                    "training_seed": int(training_seed),
                    "run_id": run_result["run_id"],
                    "manifest_path": run_result["manifest_path"],
                    "checkpoint_path": run_result["checkpoint_path"],
                    "registry_uri": run_result["registry_uri"],
                    "evaluation_cases": {
                        case_id: {
                            "metrics": case["metrics"],
                            "episodes": case["episodes"],
                            "reward_version": case["reward_version"],
                        }
                        for case_id, case in q_cases.items()
                    },
                })
            final_runs[candidate_id] = runs

    final_summaries = {candidate_id: summarize_finalist(runs) for candidate_id, runs in final_runs.items()}
    result = {
        "comparison_id": comparison_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": ["success_rate desc", "catastrophic_failure_rate asc", "mean_steps asc", "battery_remaining desc", "inference_latency asc", "candidate_id asc"],
        "evaluation_suite_id": suite["evaluation_suite_id"],
        "evaluation_suite_version": suite["version"],
        "evaluation_suite_path": str(suite_path.relative_to(PROJECT_ROOT)),
        "evaluation_suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "evaluation_seeds": [int(seed) for seed in suite["seeds"]],
        "evaluation_episode_count": int(suite["episodes_per_level"]),
        "evaluation_case_count": len(suite["levels"]),
        "reward_version": reward_version,
        "screening_seed": int(screen_seed),
        "screening_budget": int(screening_budget),
        "final_training_budget": final_budget,
        "training_seeds": [int(seed) for seed in training_seeds],
        "screening": {
            "candidates": screening,
            "ranking": ranked,
            "selected_finalists": selected_ids,
            "rejected_candidates": [item["candidate_id"] for item in ranked[2:]],
        },
        "finalists": final_summaries,
        "return_policy": "No return aggregation or ranking across cases; each case includes its own reward_version.",
        "variance_policy": "Between-training-seed SD is computed over each model's mean across common suite episodes. Evaluation-episode SD is computed within each trained model then averaged across the three models.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    decision.write_text(decision_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--screening-episodes", type=int, default=DEFAULT_SCREENING_EPISODES)
    parser.add_argument("--screen-seed", type=int, default=DEFAULT_SCREEN_SEED)
    parser.add_argument("--training-seeds", nargs=3, type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--final-episodes", type=int, help="Por defecto usa el presupuesto común de configs/experiments")
    parser.add_argument("--output", help="JSON de comparación; el nombre debe ser nuevo")
    parser.add_argument("--decision", default=str(DEFAULT_DECISION_PATH))
    args = parser.parse_args()
    result = run_comparison(
        candidates=tuple(args.candidates),
        screening_budget=args.screening_episodes,
        training_seeds=tuple(args.training_seeds),
        screen_seed=args.screen_seed,
        final_budget=args.final_episodes,
        output_path=args.output,
        decision_path=args.decision,
    )
    print(json.dumps({
        "comparison_id": result["comparison_id"],
        "finalists": result["screening"]["selected_finalists"],
        "rejected": result["screening"]["rejected_candidates"],
        "episodes_per_finalist": result["final_training_budget"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
