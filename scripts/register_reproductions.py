"""Registra en el MLflow del proyecto las reproducciones de exp-001..004 (scripts/reproduce_exp.py).

Cada reproducción se entrenó con su propio MLflow en una carpeta aparte. Este script copia sus
parámetros, métricas y artefactos al experimento drone-rescue-reproducciones, enlazados al run
original con la etiqueta reproduction_of, y guarda config, normalización y evaluación en
artifacts/results/reproducciones/<candidato>-seed-<semilla>/.

    python scripts/register_reproductions.py <carpeta con una subcarpeta por reproducción>
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from migrate_mlflow_to_remote import _copy_artifacts, _copy_metrics, _copy_params  # noqa: E402
from pipelines.tracking import ARTIFACT_ROOT, DEFAULT_TRACKING_URI  # noqa: E402

EXPERIMENT = "drone-rescue-reproducciones"
SUMMARY = ROOT / "artifacts" / "results" / "reproduccion_exp_001_004.json"
OUTPUT = ROOT / "artifacts" / "results" / "reproducciones"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(source_root: Path) -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    destination = MlflowClient(DEFAULT_TRACKING_URI)
    experiment = destination.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else destination.create_experiment(
        EXPERIMENT, artifact_location=f"{ARTIFACT_ROOT}/{EXPERIMENT}")
    for row in summary["runs"]:
        name = f"{row['candidato']}-seed-{row['semilla']}"
        folder = source_root / f"{row['candidato']}-{row['semilla']}"  # como la escribe reproduce_exp.py en el ejemplo
        source = MlflowClient(f"sqlite:///{folder / 'mlflow.db'}")
        manifest = json.loads(next((folder / "manifests").glob("*.json")).read_text(encoding="utf-8"))
        if manifest["model_sha256"] != row["sha256_reproducido"]:
            raise ValueError(f"{name}: la reproducción no coincide con la registrada antes")
        existing = destination.search_runs([experiment_id], filter_string=f"tags.reproduction_of = '{row['run_original']}'")
        if existing:
            row["mlflow_run_id"] = existing[0].info.run_id
            print(f"{name}: ya registrado como {row['mlflow_run_id']}")
            continue
        run = source.get_run(manifest["run_id"])
        tags = {key: value for key, value in run.data.tags.items() if not key.startswith("mlflow.")}
        tags.update({"reproduction_of": row["run_original"], "source_run_id": run.info.run_id, "provenance": "reproduccion",
                     "bit_identical": str(row["bit_a_bit"]).lower(),
                     "reproduction_note": "pesos originales perdidos; reentrenado con el mismo commit, perfil, semilla y suite"})
        remote = destination.create_run(experiment_id, start_time=run.info.start_time, run_name=f"{name}-reproduccion", tags=tags)
        _copy_params(run, destination, remote.info.run_id)
        _copy_metrics(source, destination, run.info.run_id, remote.info.run_id)
        _copy_artifacts(source, destination, run.info.run_id, remote.info.run_id)
        destination.set_terminated(remote.info.run_id, status=run.info.status, end_time=run.info.end_time)
        row["mlflow_run_id"] = remote.info.run_id

        target = OUTPUT / name
        target.mkdir(parents=True, exist_ok=True)
        for artifact in ("config/config.json", "normalization/normalization.json", "evaluation/evaluacion.json"):
            downloaded = Path(source.download_artifacts(run.info.run_id, artifact, str(target)))
            shutil.move(str(downloaded), target / downloaded.name)
            shutil.rmtree(target / artifact.split("/")[0], ignore_errors=True)
        row["artefactos"] = {path.name: sha256(path) for path in sorted(target.iterdir())}
        print(f"{name}: {run.info.run_id} -> {remote.info.run_id}")
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    main(Path(sys.argv[1]).resolve())
