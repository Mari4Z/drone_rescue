import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main as api


RUN_ID = "test-run-001"


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    runtime = api.ApiRuntime()
    monkeypatch.setattr(api, "runtime", runtime)
    monkeypatch.setattr(api, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(api, "EVALUATION_SUITE_PATH", api.PROJECT_ROOT / "configs" / "finalist_evaluation_suite.yaml")
    (tmp_path / "manifests").mkdir()

    environment = api.DroneEnv()
    resolved_config = environment.config
    resolved_config["experiment"] = {
        "model_family": "drone-rescue-v2",
        "changed_dimension": "world.obstacles",
    }
    resolved_config["world"]["obstacles"] = [
        {"min_m": [30.0, 30.0, 0.0], "max_m": [35.0, 35.0, 10.0]}
    ]
    environment.close()
    config_path = tmp_path / "config.json"
    normalization_path = tmp_path / "normalization.json"
    q_table_path = tmp_path / "q_table.npy"
    evaluation_path = tmp_path / "evaluation.json"
    config_path.write_text(json.dumps({"environment": resolved_config}), encoding="utf-8")
    normalization_path.write_text(json.dumps({
        "action_dimensions": 7,
        "family": "drone-rescue-v2",
        "method": "discretize_observation",
        "parameters": resolved_config["q_learning"]["discretization"],
        "state_dimensions": 16,
    }), encoding="utf-8")
    table = np.empty(0, dtype=[("states", np.int64, (16,)), ("q_values", np.float64, (7,))])
    np.save(q_table_path, table, allow_pickle=False)
    evaluation_path.write_text(json.dumps({
        "cases": [
            {"metrics": {"mean_steps": 12.0}},
            {"metrics": {"mean_steps": 8.0}},
        ],
    }), encoding="utf-8")
    artifact_paths = {
        "model": q_table_path,
        "config": config_path,
        "normalization": normalization_path,
        "evaluation": evaluation_path,
    }
    manifest = {
        "run_id": RUN_ID,
        "level_id": "N1",
        "git_commit": "abc123",
        "environment_version": api.ENVIRONMENT_VERSION,
        "evaluation_suite_id": "drone-rescue-finalists-public",
        "evaluation_suite_version": "1.0.0",
        "registry_version": "3",
        "evaluation_uri": "test:/evaluation",
        "evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        "evaluation_suite_sha256": hashlib.sha256(
            api.EVALUATION_SUITE_PATH.read_bytes()
        ).hexdigest(),
    }
    for key, path in artifact_paths.items():
        manifest[f"{key}_uri"] = f"test:/{key}"
        manifest[f"{key}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifests" / f"{RUN_ID}.json").write_text(json.dumps(manifest), encoding="utf-8")

    def download_artifact(artifact_uri):
        artifact_name = artifact_uri.rsplit("/", 1)[-1]
        if artifact_uri.endswith("/evaluation/evaluacion.json"):
            return str(artifact_paths["evaluation"])
        return str(artifact_paths[artifact_name])

    monkeypatch.setattr(api.mlflow.artifacts, "download_artifacts", download_artifact)
    return TestClient(api._create_app(production=False, cors_origins=[]))


def _load(client):
    response = client.post("/load", json={"run_id": RUN_ID, "seed": 17})
    assert response.status_code == 200
    return response


def test_health_and_version(api_client):
    assert api_client.get("/health").json() == {"status": "ok", "model_loaded": False}
    version = api_client.get("/version")
    assert version.status_code == 200
    assert version.json() == {
        "api_version": api.API_VERSION,
        "environment_version": api.ENVIRONMENT_VERSION,
    }


def test_catalog_joins_mlflow_runs_and_manifests(api_client, monkeypatch):
    class RunInfo:
        run_id = RUN_ID
        status = "FINISHED"
        experiment_id = "1"
        start_time = 1_000
        end_time = 2_500

    class RunData:
        tags = {"hypothesis": "test hypothesis", "git_commit": "abc123", "level_id": "N1"}
        params = {"seed": "17", "episodes": "500", "experiment_id": "exp-001"}
        metrics = {"success_rate": 0.5}

    class Run:
        info = RunInfo()
        data = RunData()

    class Page(list):
        token = None

    class Experiment:
        experiment_id = "1"
        name = "test"

    class Client:
        def search_experiments(self):
            return [Experiment()]

        def search_runs(self, **kwargs):
            return Page([Run()])

    monkeypatch.setattr(api, "MlflowClient", Client)
    response = api_client.get("/catalog")
    assert response.status_code == 200
    run = response.json()["runs"][0]
    assert run["run_id"] == RUN_ID
    assert run["status"] == "FINISHED"
    assert run["hypothesis"] == "test hypothesis"
    assert run["commit"] == "abc123"
    assert run["level"] == "N1"
    assert run["experiment_id"] == "exp-001"
    assert run["mlflow_experiment"] == "test"
    assert run["metrics"] == {"success_rate": 0.5}
    assert run["manifest_available"] is True
    assert run["mlflow_link"] is None
    assert run["mlflow_run_uri"] == f"runs:/{RUN_ID}"
    assert run["seed"] == 17
    assert run["training_episodes"] == 500
    assert run["duration_seconds"] == 1.5
    assert run["mean_steps"] == 10.0
    assert run["environment_version"] == api.ENVIRONMENT_VERSION
    assert run["model_version"] == "3"
    assert run["model_available"] is True
    assert run["changed_dimension"] == "world.obstacles"
    assert {change["path"] for change in run["changes_vs_baseline"]} == {"world.obstacles"}
    assert run["config_error"] is None
    assert run["evaluation_error"] is None


def test_load_returns_active_model_environment_and_config(api_client):
    response = _load(api_client)
    payload = response.json()
    assert payload["model"]["run_id"] == RUN_ID
    assert payload["model"]["sha256"]
    assert payload["environment"]["version"] == api.ENVIRONMENT_VERSION
    assert payload["config"]["q_learning"]["discretization"]["battery_fraction_bins"] == 5
    assert payload["config"]["world"]["obstacles"] == [
        {"min_m": [30.0, 30.0, 0.0], "max_m": [35.0, 35.0, 10.0]}
    ]
    assert payload["config_sha256"]
    assert payload["mode"] == "manual"


def test_load_rejects_hash_mismatch(api_client):
    manifest_path = api.MANIFEST_DIR / f"{RUN_ID}.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    response = api_client.post("/load", json={"run_id": RUN_ID})
    assert response.status_code == 409
    assert "SHA-256" in response.json()["detail"]


def test_load_rejects_unsupported_environment_version(api_client):
    manifest_path = api.MANIFEST_DIR / f"{RUN_ID}.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["environment_version"] = "1.0.0"
    manifest_path.write_text(json.dumps(manifest))
    response = api_client.post("/load", json={"run_id": RUN_ID})
    assert response.status_code == 409
    assert "adaptador histórico" in response.json()["detail"]


def test_evaluate_requires_loaded_model_and_executes_suite(api_client, monkeypatch):
    assert api_client.post("/evaluate").status_code == 409
    _load(api_client)
    result = {"cases": [{"case_id": "N1-q_learning_greedy"}]}
    monkeypatch.setattr(api, "evaluate", lambda suite, model, output: result)
    response = api_client.post("/evaluate")
    assert response.status_code == 200
    assert response.json()["result"] == result
    assert response.json()["result_path"].startswith("artifacts/results/api-evaluation-")


def test_episode_manual_step_reports_state_telemetry_reward_cause_and_mode(api_client):
    _load(api_client)
    response = api_client.get("/episode/step", params={"action": 4})
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "manual"
    assert len(payload["position_m"]) == 3
    assert "battery_wh" in payload["telemetry"]
    assert isinstance(payload["reward"], float)
    assert "cause" in payload
    assert payload["telemetry"]["steps"] == 1


def test_episode_reset_random_and_reference_modes(api_client):
    _load(api_client)
    random_response = api_client.post("/episode/reset", json={"mode": "aleatorio", "seed": 7})
    assert random_response.status_code == 200
    assert random_response.json()["mode"] == "aleatorio"
    assert api_client.get("/episode/step").status_code == 200

    reference_response = api_client.post("/episode/reset", json={"mode": "referencia", "seed": 8})
    assert reference_response.status_code == 200
    assert api_client.get("/episode/step").status_code == 200

    evaluation_response = api_client.post("/episode/reset", json={"mode": "evaluacion", "seed": 9})
    assert evaluation_response.status_code == 200
    assert api_client.get("/episode/step").status_code == 200


@pytest.mark.parametrize("mode", ["entrenamiento", "replay"])
def test_unimplemented_modes_return_real_http_errors(api_client, mode):
    _load(api_client)
    response = api_client.post("/episode/reset", json={"mode": mode})
    assert response.status_code == 501
    assert mode in response.json()["detail"]


def test_episode_requires_manual_action_and_rejects_mode_switch(api_client):
    _load(api_client)
    assert api_client.get("/episode/step").status_code == 422
    assert api_client.get("/episode/step", params={"action": 7}).status_code == 422
    assert api_client.get("/episode/step", params={"mode": "replay"}).status_code == 409


def test_production_requires_https_and_https_cors_origins():
    with pytest.raises(RuntimeError, match="CORS"):
        api._create_app(production=True, cors_origins=["http://localhost:5173"])
    secure_client = TestClient(api._create_app(production=True, cors_origins=[]))
    assert secure_client.get("/health").status_code == 426
    https_client = TestClient(
        api._create_app(production=True, cors_origins=[]),
        base_url="https://testserver",
    )
    assert https_client.get("/health").status_code == 200


def test_cors_uses_configured_origins():
    client = TestClient(api._create_app(production=False, cors_origins=["https://web.example"]))
    response = client.options(
        "/health",
        headers={
            "Origin": "https://web.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://web.example"
