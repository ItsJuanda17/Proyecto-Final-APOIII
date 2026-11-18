FROM python:3.11-slim

# Librerías del sistema que necesita OpenCV (instalado por mediapipe)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/best_model.pkl \
    SAMPLES_DIR=/app/samples

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY ["Entrega 3/src/", "src/"]
COPY models/best_model.pkl models/best_model.pkl
COPY samples/ samples/

CMD ["sh", "-c", "uvicorn src.web:app --host 0.0.0.0 --port ${PORT:-8000}"]
