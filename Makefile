RUN ?= $(shell ls -td artifacts/*/ 2>/dev/null | head -1)

.PHONY: install test ingest train evaluate export serve docker all

install:
	pip install -e ".[train,dev]"

test:
	pytest -q

ingest:
	python -m defectinspect.data.ingest --config configs/default.yaml

train:
	python -m defectinspect.train --config configs/default.yaml

evaluate:
	python -m defectinspect.evaluate --run $(RUN)

export:
	python -m defectinspect.export --run $(RUN)

serve:
	MODEL_DIR=$(RUN) uvicorn defectinspect.api.app:app --port 8000

docker:
	rm -rf model && mkdir model && cp $(RUN)/model.onnx $(RUN)/labels.json model/
	docker compose up --build

all: ingest train evaluate export
