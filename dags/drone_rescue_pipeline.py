"""DAG del pipeline: validar, probar, entrenar, evaluar, registrar, seleccionar, publicar y verificar.

Airflow vive en su propio entorno (.venv-airflow, Python 3.12). Cada tarea ejecuta el código del
proyecto con el entorno .venv, así el orquestador no altera las dependencias fijadas de los modelos.
Para usar un run existente en vez de entrenar, pasa el parámetro run_id.
"""

from __future__ import annotations

from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

ROOT = Path(__file__).resolve().parents[1]
STEP = f"cd {ROOT} && MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python -m pipelines.pipeline_steps"
STATE = "--state artifacts/results/pipeline/{{ run_id | replace(':', '-') | replace('+', '-') }}.json"

with DAG(
    dag_id="drone_rescue_pipeline",
    description="Sección 14 de la guía: del perfil versionado al modelo publicado y comprobado",
    schedule=None,
    catchup=False,
    params={
        "profile": Param("configs/training/ppo_v3_n1.yaml", type="string"),
        "seed": Param(11, type="integer"),
        "timesteps": Param(0, type="integer", description="0 usa el presupuesto del perfil"),
        "run_id": Param("", type="string", description="Run existente; si se indica, no entrena"),
    },
    tags=["drone-rescue", "mlops"],
) as dag:
    validate = BashOperator(task_id="1_validar", bash_command=f"{STEP} validate {STATE} --profile {{{{ params.profile }}}}")
    tests = BashOperator(task_id="2_probar", bash_command=f"{STEP} tests")
    train = BashOperator(
        task_id="3_entrenar",
        bash_command=(
            f"{STEP} train {STATE} --profile {{{{ params.profile }}}} --seed {{{{ params.seed }}}} "
            "--timesteps {{ params.timesteps }} --run-id '{{ params.run_id }}'"
        ),
    )
    evaluate = BashOperator(task_id="4_evaluar", bash_command=f"{STEP} evaluate {STATE}")
    register = BashOperator(task_id="5_registrar", bash_command=f"{STEP} register {STATE}")
    select = BashOperator(task_id="6_seleccionar", bash_command=f"{STEP} select {STATE}")
    publish = BashOperator(task_id="7_publicar", bash_command=f"{STEP} publish {STATE}")
    verify = BashOperator(task_id="8_verificar", bash_command=f"{STEP} verify {STATE}")

    validate >> tests >> train >> evaluate >> register >> select >> publish >> verify
