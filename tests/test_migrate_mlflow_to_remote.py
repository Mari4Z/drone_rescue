import json
from types import SimpleNamespace

import pytest

from scripts.migrate_mlflow_to_remote import _resolve_existing_attempts, _rewrite_manifest


def test_rewrite_manifest_points_artifacts_to_imported_remote_run(tmp_path):
    source_run_id = "source-run-1"
    remote_run_id = "remote-run-2"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "run_id": source_run_id,
        "model_uri": f"runs:/{source_run_id}/checkpoints/model.npy",
        "config_uri": f"runs:/{source_run_id}/config/config.json",
        "normalization_uri": f"runs:/{source_run_id}/normalization/normalization.json",
        "evaluation_sha256": "abc",
        "registry_uri": "models:/legacy/1",
    }))

    _rewrite_manifest(manifest_path, source_run_id, remote_run_id)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == source_run_id
    assert manifest["remote_run_id"] == remote_run_id
    assert manifest["model_uri"] == f"runs:/{remote_run_id}/checkpoints/model.npy"
    assert manifest["config_uri"] == f"runs:/{remote_run_id}/config/config.json"
    assert manifest["normalization_uri"] == f"runs:/{remote_run_id}/normalization/normalization.json"
    assert manifest["evaluation_uri"] == f"runs:/{remote_run_id}/evaluation/evaluacion.json"
    assert manifest["registry_uri"] is None


def _attempt(run_id, status):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(tags={"migration_status": status}),
    )


class FakeDestination:
    def __init__(self):
        self.tags = []
        self.terminated = []

    def set_tag(self, run_id, key, value):
        self.tags.append((run_id, key, value))

    def set_terminated(self, run_id, status):
        self.terminated.append((run_id, status))


def test_reuses_completed_attempt_without_mutating_other_attempts():
    destination = FakeDestination()
    existing = [_attempt("partial", "RUNNING"), _attempt("complete", "COMPLETED")]

    completed_id = _resolve_existing_attempts(
        destination, existing, "source-run", retry_incomplete=True,
    )

    assert completed_id == "complete"
    assert destination.tags == []
    assert destination.terminated == []


def test_incomplete_attempt_requires_explicit_recovery():
    destination = FakeDestination()

    with pytest.raises(RuntimeError, match="--retry-incomplete"):
        _resolve_existing_attempts(
            destination, [_attempt("partial", "RUNNING")], "source-run",
            retry_incomplete=False,
        )

    assert destination.tags == []
    assert destination.terminated == []


def test_explicit_recovery_preserves_incomplete_attempt_as_failed():
    destination = FakeDestination()

    completed_id = _resolve_existing_attempts(
        destination, [_attempt("partial", "RUNNING")], "source-run",
        retry_incomplete=True,
    )

    assert completed_id is None
    assert destination.tags == [
        ("partial", "migration_status", "FAILED"),
        ("partial", "migration_recovered", "true"),
    ]
    assert destination.terminated == [("partial", "FAILED")]


def test_failed_attempt_can_be_retried_without_recovery_flag():
    destination = FakeDestination()

    completed_id = _resolve_existing_attempts(
        destination, [_attempt("failed", "FAILED")], "source-run",
        retry_incomplete=False,
    )

    assert completed_id is None
    assert destination.tags == []
    assert destination.terminated == []
