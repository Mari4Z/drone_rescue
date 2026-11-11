"""Reentrena un run de exp-001..004 con el código original para recuperar sus pesos.

Los pesos de esos runs (5 de noviembre) solo existían en el equipo donde se
entrenaron. Este script repite el entrenamiento con el mismo commit, perfil,
semilla, presupuesto y suite, y escribe un MLflow, manifiesto y tabla Q propios
en el directorio de salida, sin tocar el proyecto.

Se ejecuta desde una copia del repositorio en el commit original:

    git worktree add --detach ../wt-1b9fa6d 1b9fa6d
    cd ../wt-1b9fa6d
    python /ruta/al/proyecto/scripts/reproduce_exp.py exp-001 11 5000 /tmp/repro/exp-001-11

Cribado: semilla 7001 y 500 episodios. Finalistas (exp-001 y exp-002): semillas
11, 23 y 37 con 5000 episodios. Resultados: artifacts/results/reproduccion_exp_001_004.json.
"""

import json
import os
import sys
from pathlib import Path

candidate, seed, episodes, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]).resolve()
out.mkdir(parents=True, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{out}/mlflow.db"
sys.path.insert(0, os.getcwd())

from pipelines.compare import PROJECT_ROOT, _load_candidate, _write_run_profile  # noqa: E402
from pipelines.evaluate import load_suite  # noqa: E402
from pipelines.train import train_run  # noqa: E402

suite, suite_path = load_suite(PROJECT_ROOT / "configs" / "finalist_evaluation_suite.yaml")
cand = _load_candidate(candidate)
profile = _write_run_profile(cand, suite, int(cand["episode_budget"]), out)
result = train_run(profile, seed, suite_path=suite_path, episodes_override=episodes,
                   manifest_dir=out / "manifests", checkpoint_dir=out / "q_tables")
manifest = json.loads(Path(result["manifest_path"]).read_text())
print(json.dumps({"candidate": candidate, "seed": seed, "run_id": manifest["run_id"],
                  "model_sha256": manifest["model_sha256"]}))
