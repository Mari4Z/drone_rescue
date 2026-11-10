import hashlib
import json
import time
from contextlib import contextmanager
from types import SimpleNamespace

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
    session_id = "00000000-0000-0000-0000-000000000001"
    response = client.post("/load", json={"run_id": RUN_ID, "seed": 17, "session_id": session_id})
    assert response.status_code == 200
    client.headers.update({"X-Drone-Session": response.json()["session_id"]})
    return response


def test_health_and_version(api_client):
    assert api_client.get("/health").json() == {"status": "ok", "model_loaded": False}
    version = api_client.get("/version")
    assert version.status_code == 200
    assert version.json()["api_version"] == api.API_VERSION
    assert version.json()["environment_version"] == api.ENVIRONMENT_VERSION
    assert version.json()["served_model"] is None


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

    class IncompleteRun:
        info = SimpleNamespace(run_id="partial-import", experiment_id="1")
        data = SimpleNamespace(
            tags={"source_run_id": RUN_ID, "migration_status": "RUNNING"},
            params={},
            metrics={},
        )

    class Page(list):
        token = None

    class Experiment:
        experiment_id = "1"
        name = "test"

    class Client:
        def search_experiments(self):
            return [Experiment()]

        def search_runs(self, **kwargs):
            return Page([IncompleteRun(), Run()])

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
    assert len(response.json()["runs"]) == 1
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
    api_client.headers.update({"X-Drone-Session": "00000000-0000-0000-0000-000000000002"})
    assert api_client.post("/evaluate").status_code == 409
    _load(api_client)
    result = {"cases": [{"case_id": "N1-q_learning_greedy"}]}
    monkeypatch.setattr(api, "evaluate", lambda suite, model, output: result)
    monkeypatch.setattr(api, "MlflowClient", lambda: SimpleNamespace(
        get_run=lambda run_id: SimpleNamespace(info=SimpleNamespace(experiment_id="1"))
    ))

    @contextmanager
    def fake_start_run(**kwargs):
        yield SimpleNamespace(info=SimpleNamespace(run_id="evaluation-run-001"))

    monkeypatch.setattr(api.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(api.mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(api.mlflow, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(api.mlflow, "log_metric", lambda *args, **kwargs: None)
    response = api_client.post("/evaluate")
    assert response.status_code == 200
    assert response.json()["result"] == result
    assert response.json()["result_path"] == "runs:/evaluation-run-001/evaluation/evaluacion.json"


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


def test_training_mode_requires_training_endpoint(api_client):
    response = api_client.post("/episode/reset", json={"mode": "entrenamiento"})
    assert response.status_code == 409
    assert "POST /training" in response.json()["detail"]


def test_manual_session_works_without_loading_model(api_client):
    response = api_client.post("/session", json={"config_id": "baseline", "seed": 11})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["session_id"]) == 36
    assert payload["state"]["mode"] == "manual"
    assert payload["environment"]["version"] == api.ENVIRONMENT_VERSION
    api_client.headers.update({"X-Drone-Session": payload["session_id"]})

    step = api_client.get("/episode/step", params={"action": 4})
    assert step.status_code == 200
    assert step.json()["run_id"] == "manual-baseline"


def test_multiple_clients_keep_independent_simulation_sessions(api_client):
    first = api_client.post("/session", json={"config_id": "baseline", "seed": 1}).json()
    api_client.headers.update({"X-Drone-Session": first["session_id"]})
    assert api_client.get("/episode/step", params={"action": 4}).json()["telemetry"]["steps"] == 1

    second = api_client.post("/session", json={"config_id": "n1", "seed": 2}).json()
    api_client.headers.update({"X-Drone-Session": second["session_id"]})
    assert api_client.get("/episode/step", params={"action": 4}).json()["telemetry"]["steps"] == 1

    api_client.headers.update({"X-Drone-Session": first["session_id"]})
    assert api_client.post("/episode/reset", json={"mode": "manual", "seed": 3}).json()["telemetry"]["steps"] == 0
    api_client.headers.update({"X-Drone-Session": second["session_id"]})
    assert api_client.get("/episode/step", params={"action": 4}).json()["telemetry"]["steps"] == 2


def test_replay_requires_model_in_manual_session(api_client):
    response = api_client.post("/session", json={"config_id": "baseline"})
    assert response.status_code == 200
    api_client.headers.update({"X-Drone-Session": response.json()["session_id"]})
    response = api_client.post("/episode/reset", json={"mode": "replay", "seed": 1})
    assert response.status_code == 409
    assert "modelo" in response.json()["detail"]


def test_replay_repeats_same_policy_trajectory_for_seed(api_client):
    _load(api_client)
    first_reset = api_client.post("/episode/reset", json={"mode": "replay", "seed": 23})
    assert first_reset.status_code == 200
    first_steps = [api_client.get("/episode/step").json() for _ in range(3)]

    second_reset = api_client.post("/episode/reset", json={"mode": "replay", "seed": 23})
    assert second_reset.status_code == 200
    second_steps = [api_client.get("/episode/step").json() for _ in range(3)]
    assert first_steps == second_steps


def test_training_job_reports_progress_and_run_id(api_client, monkeypatch):
    def fake_train_run(profile_path, seed, *, episodes_override, progress_callback):
        progress_callback({
            "episode": 1,
            "total_episodes": episodes_override or 2,
            "return": -1.0,
            "success": False,
            "epsilon": 0.9,
            "cause": "collision",
            "steps": 3,
        })
        return {
            "run_id": "trained-run-1",
            "manifest_path": "manifests/trained-run-1.json",
            "checkpoint_path": "artifacts/q_tables/trained-run-1.npy",
        }

    monkeypatch.setattr(api, "train_run", fake_train_run)
    response = api_client.post("/training", json={"profile_id": "n1", "seed": 7, "episodes": 2})
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    for _ in range(50):
        status = api_client.get(f"/training/{job_id}")
        if status.json()["status"] != "RUNNING":
            break
        time.sleep(0.01)
    payload = status.json()
    assert payload["status"] == "FINISHED"
    assert payload["run_id"] == "trained-run-1"
    assert payload["episode"] == 1


def test_episode_requires_manual_action_and_rejects_mode_switch(api_client):
    _load(api_client)
    assert api_client.get("/episode/step").status_code == 422
    assert api_client.get("/episode/step", params={"action": 7}).status_code == 422
    assert api_client.get("/episode/step", params={"mode": "replay"}).status_code == 409


def test_production_requires_https_and_https_cors_origins():
    with pytest.raises(RuntimeError, match="CORS"):
        api._create_app(production=True, cors_origins=["http://localhost:5173"])
    secure_client = TestClient(api._create_app(production=True, cors_origins=[], api_token="test-token"))
    assert secure_client.get("/health").status_code == 426
    https_client = TestClient(
        api._create_app(production=True, cors_origins=[], api_token="test-token"),
        base_url="https://testserver",
    )
    assert https_client.get("/health").status_code == 200


def test_production_api_requires_shared_token(api_client):
    secured = TestClient(
        api._create_app(production=True, cors_origins=[], api_token="secret"),
        base_url="https://testserver",
    )
    assert secured.get("/catalog").status_code == 401
    response = secured.post(
        "/session",
        headers={"X-API-Key": "secret"},
        json={"config_id": "baseline", "seed": 2},
    )
    assert response.status_code == 200


def test_cors_uses_configured_origins():
    client = TestClient(
        api._create_app(production=False, cors_origins=["https://web.example"], api_token="secret")
    )
    response = client.options(
        "/health",
        headers={
            "Origin": "https://web.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://web.example"
    unauthorized = client.get("/catalog", headers={"Origin": "https://web.example"})
    assert unauthorized.status_code == 401
    assert unauthorized.headers["access-control-allow-origin"] == "https://web.example"
