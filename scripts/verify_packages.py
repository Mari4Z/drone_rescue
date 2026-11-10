"""Comprueba que cada paquete de artifacts/models coincide con el SHA-256 de su manifiesto.

Úsalo después de recuperar los modelos (dvc pull o la release de GitHub):
    python scripts/verify_packages.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    problems, checked = [], 0
    for manifest_path in sorted((ROOT / "manifests").glob("[0-9a-f]*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("model_sha256"):
            continue  # run sin checkpoint: se conserva el manifiesto con su diagnóstico
        model_name = Path(manifest["model_uri"]).name
        if not manifest.get("package_path"):
            # Q-learning: el checkpoint se guarda en artifacts/q_tables/<run_id>.npy
            checks = [(ROOT / "artifacts" / "q_tables" / model_name, manifest["model_sha256"])]
            for path, expected in checks:
                checked += 1
                if not path.is_file():
                    problems.append(f"falta {path.relative_to(ROOT)} (puede ser de otro equipo: ver docs/despliegue.md)")
                elif sha256(path) != expected:
                    problems.append(f"SHA-256 distinto en {path.relative_to(ROOT)}")
            continue
        package = ROOT / manifest["package_path"]
        checks = [(package / model_name, manifest["model_sha256"])]
        if manifest.get("normalization_sha256"):
            checks.append((package / "normalization.json", manifest["normalization_sha256"]))
        if manifest.get("evaluation_sha256"):
            checks.append((package / "evaluacion.json", manifest["evaluation_sha256"]))
        for path, expected in checks:
            checked += 1
            if not path.is_file():
                problems.append(f"falta {path.relative_to(ROOT)}")
            elif sha256(path) != expected:
                problems.append(f"SHA-256 distinto en {path.relative_to(ROOT)}")
    print(f"{checked} archivos comprobados, {len(problems)} problemas")
    for problem in problems:
        print(" -", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
