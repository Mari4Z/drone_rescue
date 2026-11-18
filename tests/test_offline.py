"""La API y el pipeline funcionan con solo los manifiestos y los paquetes de la release, sin el run en MLflow.

Un docente que descarga la release en otro equipo no tiene la base de MLflow original (sus rutas de artefactos son
absolutas de la máquina donde se entrenó). Estas pruebas simulan eso: se entrena un PPO diminuto con una base de
MLflow y se sirve con otra base vacía.
"""

import json
import os
from pathlib import Path

import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from api import main as api
from pipelines import pipeline_steps as steps
from pipelines import train_ppo

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def released_run(tmp_path, monkeypatch):
    """Un run PPO entrenado con una base de MLflow y servido por la API con otra base vacía."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/original.db")
    result = train_ppo.train_run(
        os.path.join(REPO, "configs/training/ppo_v3_n1.yaml"), 3, manifest_dir=tmp_path / "manifests",
        package_dir=tmp_path / "models", timesteps_override=200, allow_dirty=True,
    )
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/sin_el_run.db")  # esta base no conoce el run
    monkeypatch.setattr(api, "runtime", api.ApiRuntime())
    monkeypatch.setattr(api, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(api, "PROJECT_ROOT", Path(REPO))
    monkeypatch.setattr(api, "PACKAGE_ROOT", tmp_path)
    client = TestClient(api._create_app(production=False, cors_origins=[], api_token=""))
    manifest = json.loads((tmp_path / "manifests" / f"{result['run_id']}.json").read_text(encoding="utf-8"))
    return client, manifest, tmp_path


def test_catalog_lists_manifest_runs_that_mlflow_does_not_know(released_run):
    client, manifest, tmp_path = released_run
    # un run de Q-learning sin paquete (su config y normalización solo viven en MLflow) y uno que falló antes de dar pesos
    (tmp_path / "manifests" / "q1.json").write_text(json.dumps({
        "run_id": "q1", "level_id": "exp-001", "git_commit": "abc", "model_sha256": "f" * 64,  # como los manifiestos de Q-learning
        "model_uri": "runs:/q1/checkpoints/q1.npy", "config_uri": "runs:/q1/config/config.json",
    }), encoding="utf-8")
    # otra Q-table cuyos pesos sí están en artifacts/q_tables/: el checkpoint existe, pero sin config no se puede servir
    (tmp_path / "artifacts" / "q_tables").mkdir(parents=True)
    (tmp_path / "artifacts" / "q_tables" / "q2.npy").write_bytes(b"pesos")
    (tmp_path / "manifests" / "q2.json").write_text(json.dumps({
        "run_id": "q2", "level_id": "N1", "model_sha256": "e" * 64, "model_uri": "runs:/q2/checkpoints/q2.npy",
    }), encoding="utf-8")
    (tmp_path / "manifests" / "fallido.json").write_text(json.dumps({
        "run_id": "fallido", "level_id": "N1", "model_sha256": None, "failure_detail": "sin checkpoint",
    }), encoding="utf-8")
    runs = {run["run_id"]: run for run in client.get("/catalog").json()["runs"]}
    entry = runs[manifest["run_id"]]
    assert entry["status"] == "FINISHED" and entry["checkpoint_status"] == "disponible"
    assert entry["model_available"] is True and entry["manifest_available"] is True
    assert entry["algorithm"] == "ppo" and entry["model_family"] == "drone-rescue-v3"
    assert entry["hypothesis"] and entry["mlflow_link"] is None and entry["model_sha256"] == manifest["model_sha256"]
    assert isinstance(entry["mean_steps"], float) and entry["training_timesteps"] == manifest["effective_timesteps"]
    assert any(key.endswith("/success_rate") for key in entry["metrics"])
    assert runs["q1"]["model_available"] is False and runs["q1"]["status"] == "FINISHED"
    assert runs["q1"]["checkpoint_status"] == "sin checkpoint disponible"  # los pesos de este run se perdieron
    assert runs["q1"]["hypothesis"]  # sin paquete, la hipótesis sale del experimento versionado en configs/
    assert runs["q2"]["checkpoint_status"] == "disponible" and runs["q2"]["model_available"] is False
    assert runs["fallido"]["status"] == "FAILED" and runs["fallido"]["checkpoint_status"] == "sin checkpoint disponible"


def test_load_and_run_from_the_local_package_with_hash_check(released_run):
    client, manifest, tmp_path = released_run
    loaded = client.post("/load", json={"run_id": manifest["run_id"], "seed": 0})
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["model"]["sha256"] == manifest["model_sha256"]
    session = {"X-Drone-Session": loaded.json()["session_id"]}
    assert client.post("/episode/reset", json={"mode": "evaluacion", "seed": 0}, headers=session).status_code == 200
    assert client.get("/episode/step", headers=session).json()["telemetry"]["steps"] == 2
    # la integridad no depende de la ruta: un archivo alterado se rechaza por su SHA-256
    model = tmp_path / "models" / manifest["run_id"] / "model.zip"
    model.write_bytes(model.read_bytes() + b"x")
    tampered = client.post("/load", json={"run_id": manifest["run_id"], "seed": 0})
    assert tampered.status_code == 409 and "SHA-256" in tampered.text


def test_load_fails_clearly_when_neither_mlflow_nor_the_package_has_the_file(released_run):
    client, manifest, tmp_path = released_run
    (tmp_path / "models" / manifest["run_id"] / "normalization.json").unlink()
    response = client.post("/load", json={"run_id": manifest["run_id"], "seed": 0})
    assert response.status_code == 502 and "paquete local" in response.text


def test_evaluate_saves_to_its_own_experiment_when_the_parent_run_is_missing(released_run, monkeypatch):
    client, manifest, _ = released_run
    loaded = client.post("/load", json={"run_id": manifest["run_id"], "seed": 0})
    session = {"X-Drone-Session": loaded.json()["session_id"]}

    def fake_evaluate(suite_path, model_path, family, output_path):  # no repite la suite completa
        Path(output_path).write_text("{}", encoding="utf-8")
        return {"cases": [{"case_id": "N1-ppo", "metrics": {"success_rate": 1.0}}]}

    monkeypatch.setattr(api, "evaluate_ppo", fake_evaluate)
    response = client.post("/evaluate", headers=session)
    assert response.status_code == 200, response.text
    assert response.json()["evaluation_run_id"]
    experiment = MlflowClient().get_experiment_by_name(api.EVALUATION_EXPERIMENT)
    assert experiment is not None
    assert len(MlflowClient().search_runs([experiment.experiment_id])) == 1


def test_pipeline_alias_is_best_effort_when_the_registry_lacks_the_model(tmp_path, capsys):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/vacio.db")
    entry = {"registry_name": "no-existe", "registry_version": "1"}
    assert steps._set_alias(entry) is False
    publication = {"action": "promocion", "run_id": "r", "model_family": "f", "registry_name": "no-existe",
                   "registry_version": "1", "registry_uri": "models:/no-existe/1", "git_commit": "c",
                   "success_rate": 1.0, "mean_steps": 300.0}
    steps.write_publication("N1", {**publication, "model_sha256": "a"}, tmp_path)
    steps.write_publication("N1", {**publication, "model_sha256": "b"}, tmp_path)
    restored = steps.rollback("N1", tmp_path)  # con alias: no falla aunque este MLflow no tenga el modelo
    assert restored["action"] == "rollback" and restored["model_sha256"] == "a"
    assert "no tiene el modelo registrado" in capsys.readouterr().err


def test_pipeline_select_and_publish_annotate_what_mlflow_could_not_do(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/vacio.db")
    manifest = {"run_id": "x", "level_id": "N1", "model_family": "drone-rescue-v3", "model_sha256": "a",
                "registry_name": "m", "registry_version": "1", "registry_uri": "models:/m/1", "git_commit": "c"}
    monkeypatch.setattr(steps, "_manifest", lambda run_id, manifest_dir=None: manifest)
    real_write = steps.write_publication
    monkeypatch.setattr(steps, "write_publication", lambda level, entry, directory=None: real_write(level, entry, tmp_path))
    monkeypatch.setattr(steps, "current_published", lambda level, directory=None: None)
    evaluation = tmp_path / "evaluacion.json"
    evaluation.write_text(json.dumps({"cases": [{"level_id": "N1", "metrics": {
        "success_rate": 1.0, "mean_steps": 200.0, "cause_counts": {}}}]}), encoding="utf-8")
    state = tmp_path / "estado.json"
    state.write_text(json.dumps({"stages": {}, "run_id": "x", "evaluation_output": str(evaluation)}), encoding="utf-8")
    steps.select(str(state))
    select = json.loads(state.read_text(encoding="utf-8"))["stages"]["select"]
    assert select["decision"] == "aceptado" and select["selection_tag"].startswith("omitida")
    steps.publish(str(state))
    publish = json.loads(state.read_text(encoding="utf-8"))["stages"]["publish"]
    assert publish["published"] is True and publish["registry_alias"].startswith("omitido")
    assert steps.published_versions("N1", tmp_path)[0]["run_id"] == "x"


def test_pipeline_register_checks_the_local_package_when_mlflow_has_no_run(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/vacio.db")
    package = tmp_path / "models" / "x"
    package.mkdir(parents=True)
    (package / "model.zip").write_bytes(b"pesos")
    manifest = {"run_id": "x", "model_uri": "runs:/x/checkpoints/model.zip", "package_path": "models/x",
                "model_sha256": train_ppo._sha256(package / "model.zip"), "registry_name": "m", "registry_version": "1",
                "registry_uri": "models:/m/1", "git_commit": "c", "git_dirty": False}
    monkeypatch.setattr(steps, "_manifest", lambda run_id, manifest_dir=None: manifest)
    monkeypatch.setattr(steps, "PROJECT_ROOT", tmp_path)
    state = tmp_path / "estado.json"
    state.write_text(json.dumps({"stages": {}, "run_id": "x"}), encoding="utf-8")
    steps.register(str(state))
    evidence = json.loads(state.read_text(encoding="utf-8"))["stages"]["register"]
    assert evidence["model_source"] == "paquete local" and evidence["registry_status"] == "sin registro en este MLflow"
    (package / "model.zip").write_bytes(b"otros pesos")  # un paquete alterado no pasa la etapa
    with pytest.raises(ValueError, match="no coincide"):
        steps.register(str(state))
