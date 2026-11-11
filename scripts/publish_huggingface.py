"""Publica en Hugging Face el modelo oficial: la publicación vigente de un nivel en manifests/published/.

Sube el modelo (model.zip de Stable-Baselines3), su contrato de entrada y salida, la configuración,
la evaluación, el manifiesto, el registro de publicación, un ejemplo de uso y la ficha del modelo.
Antes de subir comprueba cada archivo con el SHA-256 de su manifiesto; después descarga el modelo
por el tag y vuelve a comprobarlo. Cada publicación crea un tag, por ejemplo modelos-20261111b.

Usa la sesión de `hf auth login`; el token nunca pasa por este script.

    python scripts/publish_huggingface.py --repo Mari4Z/drone-rescue --tag <tag> --dry-run
    python scripts/publish_huggingface.py --repo Mari4Z/drone-rescue --tag <tag>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "manifests" / "published"
GITHUB = "https://github.com/Mari4Z/drone_rescue"
# Archivos que maneja este script en el repositorio de Hugging Face; el resto (.gitattributes) no se toca.
MANAGED = ["*.md", "*.json", "*.zip", "*.py", "*.npy", "*.pt", "modelos/**", "publicados/**"]

EXAMPLE = '''"""Carga el modelo oficial de Drone Rescue desde Hugging Face y vuela una misión de N1.

Requiere el repositorio del proyecto (ambiente y adaptador de observaciones):
    git clone {github} && cd drone_rescue
    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python usar_modelo.py
"""

from huggingface_hub import hf_hub_download
from stable_baselines3 import PPO

from agents.ppo import make_env

ruta = hf_hub_download("{repo}", "model.zip", revision="{tag}")
modelo = PPO.load(ruta, device="cpu")

# Mismo ambiente con el que se entrenó: familia {family}, nivel {level}.
env = make_env("{config}", "{family}")
observacion, info = env.reset(seed=101)
terminado = truncado = False
while not (terminado or truncado):
    accion, _ = modelo.predict(observacion, deterministic=True)
    observacion, recompensa, terminado, truncado, info = env.step(int(accion))

print(info["cause"], "en", env.unwrapped.steps, "pasos")
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official(level: str) -> tuple[dict, Path, dict]:
    versions = sorted((PUBLISHED / level).glob("v*.json"))
    if not versions:
        raise FileNotFoundError(f"{level} no tiene modelo publicado")
    publication = json.loads(versions[-1].read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "manifests" / f"{publication['run_id']}.json").read_text(encoding="utf-8"))
    if manifest["model_sha256"] != publication["model_sha256"]:
        raise ValueError("La publicación y el manifiesto no apuntan al mismo modelo")
    return publication, versions[-1], manifest


def stage(staging: Path, level: str, repo: str, tag: str) -> dict:
    publication, publication_path, manifest = official(level)
    package = ROOT / manifest["package_path"]
    checks = {"model.zip": manifest["model_sha256"], "normalization.json": manifest["normalization_sha256"],
              "evaluacion.json": manifest["evaluation_sha256"]}
    for name, expected in checks.items():
        if sha256(package / name) != expected:
            raise ValueError(f"SHA-256 distinto en {package / name}")
        shutil.copyfile(package / name, staging / name)
    shutil.copyfile(package / "config.json", staging / "config.json")
    shutil.copyfile(ROOT / "manifests" / f"{manifest['run_id']}.json", staging / "manifest.json")
    shutil.copyfile(publication_path, staging / "publicacion.json")
    config = json.loads((package / "config.json").read_text(encoding="utf-8"))
    config_path = config.get("level", {}).get("config", "configs/levels/n1_mision_base.yaml")
    (staging / "usar_modelo.py").write_text(EXAMPLE.format(github=GITHUB, repo=repo, tag=tag, family=manifest["model_family"],
                                                          level=level, config=config_path), encoding="utf-8")
    evaluation = json.loads((package / "evaluacion.json").read_text(encoding="utf-8"))
    contract = json.loads((package / "normalization.json").read_text(encoding="utf-8"))
    (staging / "README.md").write_text(model_card(publication, manifest, evaluation, contract, repo, tag), encoding="utf-8")
    return {"publication": publication, "manifest": manifest}


def model_card(publication: dict, manifest: dict, evaluation: dict, contract: dict, repo: str, tag: str) -> str:
    level = publication["level"]
    rows = []
    for case in evaluation["cases"]:
        metrics = case["metrics"]
        rows.append(f"| {case['level_id']} | {metrics['success_rate']:.0%} | {metrics['mean_steps']:.0f} | "
                    f"{', '.join(f'{k}: {v}' for k, v in metrics['cause_counts'].items())} |")
    actions = ", ".join(f"`{index}` {name}" for index, name in enumerate(contract["actions"]))
    features = "\n".join(f"| {item['indices']} | {item['from']} | {item['scale']} |" for item in contract["features"])
    steps = f"{manifest['effective_timesteps']:,}".replace(",", " ")
    return f"""---
library_name: stable-baselines3
tags: [reinforcement-learning, gymnasium, stable-baselines3, ppo, drone]
language: [es]
---

# Drone Rescue: modelo oficial de {level}

Política PPO que pilota un dron de rescate: despega con un suministro, rodea las columnas, lo entrega despacio en la zona marcada y vuelve a la base sin agotar la batería. Proyecto final de MLOps de la Universidad Externado de Colombia; código, ambiente y pipeline en [{GITHUB}]({GITHUB}).

Es el modelo que el pipeline de Airflow publicó para {level} (versión de publicación v{publication['publication_version']:04d}), elegido con criterios fijados antes de ver resultados.

## El archivo del modelo

**`model.zip`** es el formato nativo de Stable-Baselines3 (un ZIP con `policy.pth`, los pesos de PyTorch, el optimizador y los hiperparámetros). Se carga con `PPO.load`; no hace falta descomprimirlo.

| Campo | Valor |
|---|---|
| Algoritmo | PPO (Stable-Baselines3 2.9.2, PyTorch) |
| Familia del ambiente | `{manifest['model_family']}` (versión del ambiente {manifest['environment_version']}, estabilizador {manifest.get('stabilizer_version', '-')}) |
| Experimento | `{manifest['experiment_id']}`, semilla {manifest['seed']}, {steps} pasos acumulados (incluye los del run padre) |
| Origen | continuación del run `{manifest['parent_run_id']}` |
| Run en MLflow | `{manifest['run_id']}` ([DagsHub](https://dagshub.com/Mari4Z/drone_rescue/experiments)) |
| Commit | [`{manifest['git_commit'][:7]}`]({GITHUB}/commit/{manifest['git_commit']}) |
| SHA-256 de `model.zip` | `{manifest['model_sha256']}` |

## Resultados en la suite pública

Suite `{manifest['evaluation_suite_id']}` v{manifest['evaluation_suite_version']}, 10 episodios por nivel. N1 no tiene viento: sus 10 semillas repiten el mismo vuelo.

| Nivel | Éxito | Pasos medios | Causas |
|---|---:|---:|---|
{chr(10).join(rows)}

El modelo es para {level}: no resuelve N2 (más columnas y menos batería) ni N3 (con viento).

## Cómo usarlo

El modelo espera la observación del ambiente de Drone Rescue ya escalada, así que se usa con el ambiente del repositorio de GitHub:

```bash
git clone {GITHUB} && cd drone_rescue
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```python
from huggingface_hub import hf_hub_download
from stable_baselines3 import PPO
from agents.ppo import make_env

ruta = hf_hub_download("{repo}", "model.zip", revision="{tag}")
modelo = PPO.load(ruta, device="cpu")

env = make_env("configs/levels/n1_mision_base.yaml", "{manifest['model_family']}")
observacion, info = env.reset(seed=101)
terminado = truncado = False
while not (terminado or truncado):
    accion, _ = modelo.predict(observacion, deterministic=True)
    observacion, recompensa, terminado, truncado, info = env.step(int(accion))
print(info["cause"])  # success
```

El mismo ejemplo está en `usar_modelo.py`. Con `revision="{tag}"` se descarga siempre esta versión; sin ella, la última publicada.

**Entrada:** {contract['state_dimensions']} valores, calculados por `make_features` a partir de los {contract['observation_dimensions']} de la observación (`normalization.json`):

| Índices | Origen | Escala |
|---|---|---|
{features}

**Salida:** una acción discreta de {contract['action_dimensions']}, en {contract['action_unit']}: {actions}.

## Archivos

| Archivo | Contenido |
|---|---|
| `model.zip` | El modelo (Stable-Baselines3) |
| `normalization.json` | Contrato de entrada y salida |
| `config.json` | Perfil de entrenamiento resuelto |
| `evaluacion.json` | Resultados por episodio en la suite |
| `manifest.json` | Commit, configuración, semilla, suite, origen y hashes |
| `publicacion.json` | Registro de la publicación (versión, fecha y métricas con que se aceptó) |
| `usar_modelo.py` | Ejemplo de uso |
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--level", default="N1")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="arma y muestra la ficha sin subir nada")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="hf-drone-rescue-") as directory:
        staging = Path(directory)
        info = stage(staging, args.level, args.repo, args.tag)
        files = sorted(p.name for p in staging.iterdir())
        print(f"Modelo oficial {args.level}: run {info['manifest']['run_id']} · archivos: {', '.join(files)}")
        if args.dry_run:
            print((staging / "README.md").read_text(encoding="utf-8"))
            return

        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        commit = api.upload_folder(repo_id=args.repo, folder_path=str(staging), repo_type="model",
                                   commit_message=f"Modelo oficial {args.level} ({args.tag})", delete_patterns=MANAGED)
        api.create_tag(args.repo, tag=args.tag, revision=commit.oid, tag_message=f"Modelo oficial {args.level} {args.tag}")
        downloaded = Path(hf_hub_download(args.repo, "model.zip", revision=args.tag))
        if sha256(downloaded) != info["manifest"]["model_sha256"]:
            raise ValueError("El modelo descargado no coincide con su manifiesto")
        print(f"Subido y verificado: https://huggingface.co/{args.repo} (commit {commit.oid[:8]}, tag {args.tag})")


if __name__ == "__main__":
    main()
