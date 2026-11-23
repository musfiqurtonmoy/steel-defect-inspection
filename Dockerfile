# Serving image: ONNX Runtime + FastAPI only. No PyTorch, so the image stays small.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model DB_URL=sqlite:////app/data/predictions.db

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd -m app && mkdir -p /app/data && chown app /app/data

# Copy an exported run folder here before building: model.onnx + labels.json
COPY model ./model

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "defectinspect.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
