import pytest

from pipelines import pipeline_steps as steps

CRITERIA = {"min_success_rate": {"N1": 0.8}, "max_mean_steps": 700, "no_regression": True}


def _evaluation(success, mean_steps=300.0):
    return {"cases": [{"level_id": "N1", "metrics": {"success_rate": success, "mean_steps": mean_steps, "cause_counts": {}}}]}


def _entry(sha, run="r", success=1.0, steps_=300.0):
    return {"action": "promocion", "run_id": run, "model_family": "drone-rescue-v3", "model_sha256": sha,
            "registry_name": "m", "registry_version": "1", "registry_uri": "models:/m/1", "git_commit": "c",
            "success_rate": success, "mean_steps": steps_}


def test_decide_accepts_and_rejects_with_reasons():
    assert steps.decide(_evaluation(1.0), "N1", CRITERIA, None)["decision"] == "aceptado"
    rejected = steps.decide(_evaluation(0.5), "N1", CRITERIA, None)
    assert rejected["decision"] == "rechazado" and "mínimo" in rejected["reasons"][0]
    assert steps.decide(_evaluation(1.0, 800.0), "N1", CRITERIA, None)["decision"] == "rechazado"


def test_decide_rejects_regression_against_current():
    current = {"run_id": "vigente", "success_rate": 1.0, "mean_steps": 250.0}
    assert steps.decide(_evaluation(1.0, 300.0), "N1", CRITERIA, current)["decision"] == "rechazado"
    assert steps.decide(_evaluation(1.0, 200.0), "N1", CRITERIA, current)["decision"] == "aceptado"


def test_decide_ignores_return():
    evaluation = _evaluation(0.0)
    evaluation["cases"][0]["metrics"]["mean_return"] = 10_000.0
    assert steps.decide(evaluation, "N1", CRITERIA, None)["decision"] == "rechazado"


def test_publication_never_overwrites_and_repeat_is_idempotent(tmp_path):
    first = steps.write_publication("N1", _entry("a"), tmp_path)
    assert first["publication_version"] == 1
    assert steps.write_publication("N1", _entry("a"), tmp_path)["already_published"] is True
    assert steps.write_publication("N1", _entry("b"), tmp_path)["publication_version"] == 2
    assert len(steps.published_versions("N1", tmp_path)) == 2


def test_rollback_restores_previous_as_new_version(tmp_path):
    steps.write_publication("N1", _entry("a", run="viejo"), tmp_path)
    steps.write_publication("N1", _entry("b", run="nuevo"), tmp_path)
    restored = steps.rollback("N1", tmp_path, apply_alias=False)
    assert restored["publication_version"] == 3 and restored["action"] == "rollback"
    assert restored["run_id"] == "viejo" and restored["restored_from_version"] == 1
    assert steps.current_published("N1", tmp_path)["model_sha256"] == "a"
    assert len(steps.published_versions("N1", tmp_path)) == 3


def test_rollback_needs_a_previous_version(tmp_path):
    steps.write_publication("N1", _entry("a"), tmp_path)
    with pytest.raises(ValueError, match="anterior"):
        steps.rollback("N1", tmp_path, apply_alias=False)
