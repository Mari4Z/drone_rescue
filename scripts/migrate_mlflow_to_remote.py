"""Copy local MLflow runs and artifacts to a remote tracking server."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path, PurePosixPath

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric, Param


def _rewrite_manifest(path: Path, source_run_id: str, remote_run_id: str, accepted_run_ids: set[str] = frozenset()) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("run_id") not in {source_run_id, *accepted_run_ids}:
        raise ValueError(f"El manifiesto no corresponde al run {source_run_id}: {path}")
    manifest["remote_run_id"] = remote_run_id
    if manifest.get("evaluation_sha256") and not manifest.get("evaluation_uri"):
        manifest["evaluation_uri"] = f"runs:/{remote_run_id}/evaluation/evaluacion.json"
    for key in ("model_uri", "config_uri", "normalization_uri", "evaluation_uri"):
        uri = manifest.get(key)
        if isinstance(uri, str):
            manifest[key] = uri.replace(f"runs:/{source_run_id}/", f"runs:/{remote_run_id}/")
    manifest["registry_uri"] = None
    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _artifact_paths(client: MlflowClient, run_id: str) -> list[str]:
    paths: list[str] = []
    pending = [""]
    while pending:
        directory = pending.pop()
        for item in client.list_artifacts(run_id, directory):
            if item.is_dir:
                pending.append(item.path)
            else:
                paths.append(item.path)
    return sorted(paths)


def _copy_artifacts(
    source: MlflowClient,
    destination: MlflowClient,
    source_run_id: str,
    remote_run_id: str,
) -> None:
    artifacts = _artifact_paths(source, source_run_id)
    manifest_paths = [path for path in artifacts if path.startswith("manifest/") and path.endswith(".json")]
    with tempfile.TemporaryDirectory(prefix=f"mlflow-migration-{source_run_id}-") as directory:
        root = Path(directory)
        for artifact_path in artifacts:
            if artifact_path in manifest_paths:
                continue
            downloaded = Path(source.download_artifacts(source_run_id, artifact_path, str(root)))
            parent = str(PurePosixPath(artifact_path).parent)
            destination.log_artifact(
                remote_run_id,
                str(downloaded),
                artifact_path=parent if parent != "." else None,
            )
        # Una reproducción conserva el manifiesto de su propio entrenamiento (tag source_run_id)
        accepted = {value for value in (source.get_run(source_run_id).data.tags.get("source_run_id"),) if value}
        for manifest_path in manifest_paths:
            downloaded = Path(source.download_artifacts(source_run_id, manifest_path, str(root)))
            _rewrite_manifest(downloaded, source_run_id, remote_run_id, accepted)
            destination.log_artifact(
                remote_run_id,
                str(downloaded),
                artifact_path=str(PurePosixPath(manifest_path).parent),
            )


def _copy_metrics(source: MlflowClient, destination: MlflowClient, source_run_id: str, remote_run_id: str) -> None:
    source_run = source.get_run(source_run_id)
    metric_names = source_run.data.metrics.keys()
    pending: list[Metric] = []
    for metric_name in metric_names:
        for item in source.get_metric_history(source_run_id, metric_name):
            pending.append(Metric(metric_name, item.value, item.timestamp, item.step))
            if len(pending) == 500:
                destination.log_batch(remote_run_id, metrics=pending)
                pending.clear()
    if pending:
        destination.log_batch(remote_run_id, metrics=pending)


def _copy_params(source_run, destination: MlflowClient, remote_run_id: str) -> None:
    pending: list[Param] = []
    for key, value in source_run.data.params.items():
        pending.append(Param(key, value))
        if len(pending) == 100:
            destination.log_batch(remote_run_id, params=pending)
            pending.clear()
    if pending:
        destination.log_batch(remote_run_id, params=pending)


def _resolve_existing_attempts(
    destination: MlflowClient,
    existing: list,
    source_run_id: str,
    *,
    retry_incomplete: bool,
) -> str | None:
    completed = next(
        (item for item in existing if item.data.tags.get("migration_status") == "COMPLETED"),
        None,
    )
    if completed:
        return completed.info.run_id

    incomplete = [
        item for item in existing
        if item.data.tags.get("migration_status") != "FAILED"
    ]
    if incomplete and not retry_incomplete:
        states = ", ".join(
            f"{item.info.run_id}: {item.data.tags.get('migration_status', 'sin estado')}"
            for item in incomplete
        )
        raise RuntimeError(
            f"El run {source_run_id} tiene intentos incompletos ({states}). "
            "Verifica que no haya otra migración en curso y repite con --retry-incomplete "
            "para conservarlos como FAILED e iniciar un intento nuevo."
        )

    for item in incomplete:
        destination.set_tag(item.info.run_id, "migration_status", "FAILED")
        destination.set_tag(item.info.run_id, "migration_recovered", "true")
        destination.set_terminated(item.info.run_id, status="FAILED")
    return None


def migrate(
    source_uri: str,
    destination_uri: str,
    *,
    dry_run: bool = False,
    retry_incomplete: bool = False,
    skip_run_ids: frozenset[str] = frozenset(),
) -> dict[str, str]:
    source = MlflowClient(tracking_uri=source_uri)
    source_runs = []
    source_experiments = source.search_experiments()
    for experiment in source_experiments:
        token = None
        while True:
            page = source.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=1000,
                page_token=token,
            )
            source_runs.extend((experiment, run) for run in page if run.info.run_id not in skip_run_ids)
            token = page.token
            if not token:
                break
    if dry_run:
        print(f"Se encontraron {len(source_runs)} runs en {len(source_experiments)} experimentos.")
        return {}

    destination = MlflowClient(tracking_uri=destination_uri)
    destination_experiments = {
        item.name: item.experiment_id for item in destination.search_experiments()
    }
    mappings: dict[str, str] = {}
    for experiment, run in source_runs:
        existing = destination.search_runs(
            experiment_ids=list(destination_experiments.values()),
            filter_string=f"tags.source_run_id = '{run.info.run_id}'",
            max_results=1000,
        ) if destination_experiments else []
        if existing:
            completed_id = _resolve_existing_attempts(
                destination,
                existing,
                run.info.run_id,
                retry_incomplete=retry_incomplete,
            )
            if completed_id:
                mappings[run.info.run_id] = completed_id
                continue
        destination_experiment_id = destination_experiments.get(experiment.name)
        if destination_experiment_id is None:
            destination_experiment_id = destination.create_experiment(experiment.name)
            destination_experiments[experiment.name] = destination_experiment_id

        tags = {
            key: value
            for key, value in run.data.tags.items()
            if not key.startswith("mlflow.")
        }
        tags["source_run_id"] = run.info.run_id
        tags["migration_status"] = "RUNNING"
        tags["migration_attempt"] = str(len(existing) + 1 if existing else 1)
        remote_run = destination.create_run(
            destination_experiment_id,
            start_time=run.info.start_time,
            run_name=run.data.tags.get("mlflow.runName"),
            tags=tags,
        )
        remote_run_id = remote_run.info.run_id
        mappings[run.info.run_id] = remote_run_id
        try:
            _copy_params(run, destination, remote_run_id)
            _copy_metrics(source, destination, run.info.run_id, remote_run_id)
            _copy_artifacts(source, destination, run.info.run_id, remote_run_id)
            destination.set_tag(remote_run_id, "migration_status", "COMPLETED")
            end_time = run.info.end_time
            status = run.info.status if run.info.status in {"FINISHED", "FAILED", "KILLED"} else "KILLED"
            destination.set_terminated(remote_run_id, status=status, end_time=end_time)
        except Exception as error:
            destination.set_tag(remote_run_id, "migration_status", "FAILED")
            destination.set_terminated(remote_run_id, status="FAILED")
            raise RuntimeError(f"No se pudo migrar el run {run.info.run_id}: {error}") from error
        print(f"{run.info.run_id} -> {remote_run_id} ({run.info.status})")
    return mappings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--destination-uri", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="Conserva intentos incompletos como FAILED y crea un intento nuevo; usar solo sin otra migración activa.",
    )
    parser.add_argument("--skip-run", action="append", default=[], help="Run local que no se migra; se puede repetir.")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.destination_uri)
    migrate(
        args.source_uri,
        args.destination_uri,
        dry_run=args.dry_run,
        retry_incomplete=args.retry_incomplete,
        skip_run_ids=frozenset(args.skip_run),
    )


if __name__ == "__main__":
    main()
