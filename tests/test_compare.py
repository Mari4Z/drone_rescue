from pipelines.compare import rank_screening_candidates, summarize_finalist


def _candidate(candidate_id, success, failures, steps=20, battery=50, latency=1.0, reward=9999):
    return {
        "candidate_id": candidate_id,
        "screening_metrics": {
            "success_rate": success,
            "cause_counts": failures,
            "episodes": 10,
            "mean_steps": steps,
            "mean_battery_remaining_wh": battery,
            "mean_inference_latency_ms": latency,
            "mean_return": reward,
        },
    }


def test_screening_uses_success_and_not_return():
    ranked = rank_screening_candidates([
        _candidate("exp-001", 0.5, {"collision": 2}, reward=100000),
        _candidate("exp-002", 0.6, {"collision": 3}, reward=-100000),
        _candidate("exp-003", 0.5, {"collision": 1}, reward=-100000),
    ])
    assert [row["candidate_id"] for row in ranked] == ["exp-002", "exp-003", "exp-001"]


def test_summary_separates_training_seed_and_episode_variance():
    runs = []
    for seed, success_outcomes in ((11, [1, 0]), (23, [1, 1]), (37, [0, 0])):
        episodes = [
            {"success": bool(success), "cause": "success" if success else "collision", "steps": i + 1,
             "battery_remaining_wh": 20 - i, "inference_latency_ms": 0.5 + i}
            for i, success in enumerate(success_outcomes)
        ]
        runs.append({"training_seed": seed, "run_id": str(seed), "manifest_path": "m.json",
                     "checkpoint_path": "q.npy", "registry_uri": "models:/x/1",
                     "evaluation_cases": {"exp-001": {"episodes": episodes, "metrics": {}}}})

    summary = summarize_finalist(runs)
    success = summary["metrics"]["success_rate"]
    assert success["between_training_seeds"]["mean"] == 0.5
    assert success["between_training_seeds"]["std"] > 0
    assert success["within_evaluation_episodes"]["mean_std"] > 0
    assert "failure:collision" in summary["metrics"]
