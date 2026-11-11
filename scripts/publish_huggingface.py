"""Publica los modelos versionados en un repositorio de modelos de Hugging Face.

Cada run con pesos (según su manifiesto en manifests/, solo los que están en Git) queda en
modelos/<experimento>/<run_id>/ con su manifiesto. El README es la ficha del modelo y cada
publicación crea un tag, por ejemplo modelos-20261111. Antes de subir, cada archivo se
compara con el SHA-256 de su manifiesto; después se descarga el modelo publicado y se
vuelve a comprobar.

Usa la sesión de `hf auth login`; el token nunca pasa por este script.

    python scripts/publish_huggingface.py --repo Mari4Z/drone-rescue --tag modelos-20261111 --dry-run
    python scripts/publish_huggingface.py --repo Mari4Z/drone-rescue --tag modelos-20261111
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPRODUCTIONS = ROOT / "artifacts" / "results" / "reproduccion_exp_001_004.json"
GITHUB = "https://github.com/Mari4Z/drone_rescue"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_manifests() -> list[Path]:
    names = subprocess.run(["git", "ls-files", "manifests/*.json"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return [ROOT / name for name in names if Path(name).parent.name == "manifests" and Path(name).name[0] in "0123456789abcdef"]


def experiment_of(manifest: dict) -> str:
    if manifest.get("experiment_id"):
        return manifest["experiment_id"]
    registry = manifest.get("registry_name") or "sin-registro"
    level = str(manifest.get("level_id", "")).lower()
    return f"qlearning-{level}" if registry.startswith("DroneRescueQTable-N") else registry.replace("DroneRescueQTable-", "")


def family_of(manifest: dict) -> str:
    return manifest.get("model_family") or "drone-rescue-v2"  # los Q-learning anteriores a v3 no la guardaban


def algorithm_of(manifest: dict) -> str:
    return manifest.get("algorithm") or ("qtable" if "QTable" in str(manifest.get("registry_name")) else "-")


def level_metrics(evaluation: dict, level: str) -> dict | None:
    for case in evaluation.get("cases", []):
        if str(case.get("level_id")) == str(level) and case.get("metrics"):
            return case["metrics"]
    if "metrics" in evaluation:
        return evaluation["metrics"]
    return None


def collect(staging: Path) -> list[dict]:
    """Copia los pesos verificados a staging y devuelve una fila por run."""
    reproductions = {}
    if REPRODUCTIONS.is_file():
        reproductions = {run["run_original"]: run for run in json.loads(REPRODUCTIONS.read_text(encoding="utf-8"))["runs"]}
    rows = []
    for manifest_path in sorted(tracked_manifests()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("model_sha256"):
            continue
        run_id = manifest["run_id"]
        destination = staging / "modelos" / experiment_of(manifest) / run_id
        destination.mkdir(parents=True)
        shutil.copyfile(manifest_path, destination / "manifest.json")
        note, metrics = "", None
        model_name = Path(manifest["model_uri"]).name
        if manifest.get("package_path"):
            package = ROOT / manifest["package_path"]
            if sha256(package / model_name) != manifest["model_sha256"]:
                raise ValueError(f"SHA-256 distinto en {package / model_name}")
            for item in package.iterdir():
                shutil.copyfile(item, destination / item.name)
            evaluation = package / "evaluacion.json"
            if evaluation.is_file():
                metrics = level_metrics(json.loads(evaluation.read_text(encoding="utf-8")), manifest.get("level_id"))
        else:
            original = ROOT / "artifacts" / "q_tables" / model_name
            uploaded_sha = manifest["model_sha256"]
            if original.is_file():
                if sha256(original) != manifest["model_sha256"]:
                    raise ValueError(f"SHA-256 distinto en {original}")
                shutil.copyfile(original, destination / model_name)
            elif run_id in reproductions:
                copy = ROOT / reproductions[run_id]["archivo_reproducido"]
                uploaded_sha = reproductions[run_id]["sha256_reproducido"]
                if sha256(copy) != uploaded_sha:
                    raise ValueError(f"SHA-256 distinto en la reproducción {copy}")
                shutil.copyfile(copy, destination / "reproduccion.npy")
                (destination / "reproduccion.json").write_text(
                    json.dumps(reproductions[run_id], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                note = "reproducción (original perdido)"
            else:
                raise FileNotFoundError(f"No hay pesos para {run_id}")
            # Configuración, normalización y evaluación (§11, puntos 2, 4 y 6): las del MLflow local si
            # coinciden con el manifiesto; si el run original se perdió, las de su reproducción.
            for artifacts in (ROOT / "artifacts" / "mlflow" / "mlruns").glob(f"*/{run_id}/artifacts"):
                for name, key in (("config/config.json", "config_sha256"), ("normalization/normalization.json", "normalization_sha256"),
                                  ("evaluation/evaluacion.json", "evaluation_sha256")):
                    path = artifacts / name
                    if path.is_file() and sha256(path) == manifest.get(key):
                        shutil.copyfile(path, destination / path.name)
                evaluation = destination / "evaluacion.json"
                if evaluation.is_file():
                    metrics = level_metrics(json.loads(evaluation.read_text(encoding="utf-8")), manifest.get("level_id"))
            if run_id in reproductions:
                folder = ROOT / "artifacts" / "results" / "reproducciones" / f"{reproductions[run_id]['candidato']}-seed-{reproductions[run_id]['semilla']}"
                for path in folder.glob("*.json"):
                    if not (destination / path.name).exists():
                        shutil.copyfile(path, destination / path.name)
            if metrics is None and run_id in reproductions:
                case = reproductions[run_id]["evaluacion_reproducida"].get(manifest["level_id"])
                if case:
                    causes = case["causas"]
                    metrics = {"success_rate": causes.get("success", 0) / sum(causes.values()), "mean_steps": case["pasos_medios"]}
        level = str(manifest.get("level_id"))
        rows.append({
            "experiment": experiment_of(manifest), "run_id": run_id, "family": family_of(manifest),
            "algorithm": algorithm_of(manifest), "level": Path(level).stem if "/" in level else level, "seed": manifest.get("seed"),
            "sha256": manifest["model_sha256"] if not note else uploaded_sha,
            "provenance": manifest.get("provenance") or "registrado",
            "success": None if metrics is None else metrics.get("success_rate"),
            "steps": None if metrics is None else metrics.get("mean_steps"), "note": note,
        })
    published = ROOT / "manifests" / "published"
    if published.is_dir():
        shutil.copytree(published, staging / "publicados")
    return rows


def model_card(rows: list[dict], tag: str) -> str:
    published = sorted((ROOT / "manifests" / "published").glob("*/v*.json"))
    current = {}
    for path in published:
        entry = json.loads(path.read_text(encoding="utf-8"))
        current[entry["level"]] = entry
    lines = [
        "---",
        "library_name: stable-baselines3",
        "tags: [reinforcement-learning, gymnasium, stable-baselines3, ppo, dqn, q-learning, drone]",
        "language: [es]",
        "---",
        "",
        "# Drone Rescue: modelos versionados",
        "",
        f"Modelos de aprendizaje por refuerzo del proyecto final de MLOps (Universidad Externado de Colombia). Código, ambiente y pipeline: [{GITHUB}]({GITHUB}). Esta versión: `{tag}`.",
        "",
        "Un dron lleva un suministro desde la base hasta una zona de entrega, lo suelta despacio y vuelve, esquivando columnas y administrando la batería. Hay tres niveles: N1 (misión base), N2 (más columnas y menos batería) y N3 (N2 con viento).",
        "",
        "## Modelo publicado",
        "",
    ]
    if current:
        lines += ["| Nivel | Versión | Run | Familia | Éxito | Pasos medios |", "|---|---|---|---|---:|---:|"]
        for level, entry in sorted(current.items()):
            lines.append(f"| {level} | v{entry['publication_version']:04d} | `{entry['run_id']}` | `{entry['model_family']}` | {entry['success_rate']:.0%} | {entry['mean_steps']:.0f} |")
        lines += ["", "Lo eligió el pipeline de Airflow con criterios fijados antes de ver resultados; N2 y N3 no tienen modelo publicado porque ningún candidato alcanzó el mínimo. El historial está en `publicados/`.", ""]
    lines += [
        "## Todos los runs",
        "",
        "Se conservan también los intentos que fallaron. Cada carpeta `modelos/<experimento>/<run_id>/` trae los pesos y su `manifest.json`, con el commit, la configuración, la semilla, la suite de evaluación y el SHA-256 de los pesos.",
        "",
        "| Experimento | Algoritmo | Familia | Nivel | Semilla | Éxito | Pasos | Origen | SHA-256 |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["family"], r["experiment"], str(r["seed"]))):
        success = "-" if row["success"] is None else f"{row['success']:.0%}"
        steps = "-" if row["steps"] is None else f"{row['steps']:.0f}"
        origin = row["note"] or row["provenance"]
        lines.append(f"| `{row['experiment']}` | {row['algorithm']} | `{row['family']}` | {row['level']} | {row['seed'] if row['seed'] is not None else '-'} | {success} | {steps} | {origin} | `{row['sha256'][:12]}` |")
    lines += [
        "",
        "Las familias no son intercambiables: `drone-rescue-v3` (PPO, 11 acciones de velocidad con estabilizador), `drone-rescue-v2` (7 acciones de fuerza) y `grid-drone-v1` (cuadrícula de A a B, otra misión). Los runs históricos se entrenaron antes del registro. Los pesos originales de `exp-001` a `exp-004` se perdieron y se reentrenaron con el mismo código: el comportamiento coincide con el informe, aunque los bytes no (salvo `exp-003`).",
        "",
        "## Cómo usarlos",
        "",
        "```python",
        "from huggingface_hub import snapshot_download",
        f"ruta = snapshot_download('<este-repositorio>', revision='{tag}')",
        "```",
        "",
        f"Para cargarlos hay que usar el ambiente y los adaptadores del repositorio de GitHub (`agents/ppo.py`, `agents/qlearning.py`, `agents/grid/`), en la misma familia del modelo. Verifica cada archivo con el SHA-256 de su manifiesto (`scripts/verify_packages.py`).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="arma y revisa la carpeta sin subir nada")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="hf-drone-rescue-") as directory:
        staging = Path(directory)
        rows = collect(staging)
        (staging / "README.md").write_text(model_card(rows, args.tag), encoding="utf-8")
        files = [p for p in staging.rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files) / 1e6
        print(f"{len(rows)} runs, {len(files)} archivos, {size:.1f} MB")
        if args.dry_run:
            print((staging / "README.md").read_text(encoding="utf-8"))
            return

        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        commit = api.upload_folder(repo_id=args.repo, folder_path=str(staging), repo_type="model",
                                   commit_message=f"Modelos versionados {args.tag}", delete_patterns=["modelos/**", "publicados/**"])
        api.create_tag(args.repo, tag=args.tag, revision=commit.oid, tag_message=f"Modelos registrados {args.tag}", exist_ok=False)
        print(f"Subido: https://huggingface.co/{args.repo} (commit {commit.oid[:8]}, tag {args.tag})")

        checked = 0
        for row in rows:
            if row["note"]:
                continue
            folder = f"modelos/{row['experiment']}/{row['run_id']}"
            manifest = json.loads(Path(hf_hub_download(args.repo, f"{folder}/manifest.json", revision=args.tag)).read_text(encoding="utf-8"))
            weights = hf_hub_download(args.repo, f"{folder}/{Path(manifest['model_uri']).name}", revision=args.tag)
            if sha256(Path(weights)) != row["sha256"]:
                raise ValueError(f"El modelo descargado de {folder} no coincide con su manifiesto")
            checked += 1
        print(f"Verificado desde Hugging Face: {checked} modelos coinciden con su SHA-256")


if __name__ == "__main__":
    main()
