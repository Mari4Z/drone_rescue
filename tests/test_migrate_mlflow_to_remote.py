import json

from scripts.migrate_mlflow_to_remote import _rewrite_manifest


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
