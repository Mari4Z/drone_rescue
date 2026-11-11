# API de simulación e inferencia. El ambiente Python es la autoridad; la web vive en Vercel.
# Los modelos no van en la imagen: se montan en artifacts/ desde la release o con dvc pull.
FROM python:3.14-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DRONE_RESCUE_API_HOST=0.0.0.0 \
    DRONE_RESCUE_API_PORT=8000 \
    MLFLOW_TRACKING_URI=sqlite:///artifacts/mlflow/mlflow.db
EXPOSE 8000
CMD ["python", "-m", "api.main"]
