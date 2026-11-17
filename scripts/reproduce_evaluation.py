"""Repite la evaluación de un modelo PPO desde su manifiesto y comprueba que reproduce la registrada.

Sirve para reconstruir una evaluación a partir de la release (descomprimida en la raíz del repositorio)
o de `dvc pull`:

    python scripts/reproduce_evaluation.py              # el modelo oficial de N1 (publicación vigente)
    python scripts/reproduce_evaluation.py <run_id>     # cualquier run PPO con paquete en artifacts/models/
    python scripts/reproduce_evaluation.py --level N2   # la publicación vigente de otro nivel

Comprueba el SHA-256 del modelo contra su manifiesto, repite la suite con el modelo congelado y compara,
caso por caso y episodio por episodio, la causa de terminación y el retorno con `evaluacion.json`.
Sale con código 1 si algo no coincide o falta. No escribe nada en el repositorio.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.evaluate import DEFAULT_SUITE_PATH  # noqa: E402
from pipelines.train_ppo import _sha256, evaluate_model  # noqa: E402


def _published_run(level: str) -> str | None:
    versions = sorted((ROOT / "manifests" / "published" / level).glob("v*.json"))
    return json.loads(versions[-1].read_text(encoding="utf-8"))["run_id"] if versions else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", nargs="?", help="Run a reproducir; por omisión, la publicación vigente de --level")
    parser.add_argument("--level", default="N1", help="Nivel cuya publicación vigente se usa si no se indica run_id")
    args = parser.parse_args()

    run_id = args.run_id or _published_run(args.level)
    if not run_id:
        print(f"{args.level} no tiene un modelo publicado en manifests/published/")
        return 1
    manifest_path = ROOT / "manifests" / f"{run_id}.json"
    if not manifest_path.exists():
        print(f"No existe {manifest_path.relative_to(ROOT)}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package = ROOT / manifest.get("package_path", "")
    model, recorded_path = package / "model.zip", package / "evaluacion.json"
    if not package.is_dir():
        print(f"Falta el paquete {manifest.get('package_path')}: descomprime la release o ejecuta `dvc pull`")
        return 1
    if not model.exists():
        print("Este paquete no es un modelo PPO (no tiene model.zip). Los DQN de cuadrícula se evalúan con `cli.py eval`.")
        return 1
    if not recorded_path.exists():
        print(f"Falta {recorded_path.relative_to(ROOT)}")
        return 1
    if _sha256(model) != manifest["model_sha256"]:
        print("El SHA-256 del modelo no coincide con su manifiesto")
        return 1

    with tempfile.TemporaryDirectory() as temp:
        result = evaluate_model(DEFAULT_SUITE_PATH, model, manifest["model_family"], Path(temp) / "evaluacion.json")
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))

    print(f"Run {run_id[:8]} ({manifest.get('experiment_id')}, {manifest['model_family']}, {manifest['level_id']}): "
          f"el SHA-256 del modelo coincide con el manifiesto")
    print(f"Suite {recorded['evaluation_suite_id']} {recorded['evaluation_suite_version']}")
    same = total = 0
    ok = [case["case_id"] for case in result["cases"]] == [case["case_id"] for case in recorded["cases"]]
    for new, old in zip(result["cases"], recorded["cases"]):
        a, b = new["metrics"]["success_rate"], old["metrics"]["success_rate"]
        ok &= a == b
        print(f"  {new['case_id']}: éxito {a:.0%} (registrado {b:.0%})")
        for ep_new, ep_old in zip(new["episodes"], old["episodes"]):
            total += 1
            same += ep_new["cause"] == ep_old["cause"] and abs(ep_new["return"] - ep_old["return"]) < 1e-6
    ok &= same == total
    print(f"Episodios idénticos (causa y retorno): {same}/{total}")
    print("REPRODUCE la evaluación registrada" if ok else "NO REPRODUCE la evaluación registrada")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
