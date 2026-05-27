.PHONY: install run test lint ingest-sample

install:
	python3.10 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && uvicorn app.main:app --reload

test:
	. .venv/bin/activate && pytest

lint:
	. .venv/bin/activate && ruff check .

ingest-sample:
	. .venv/bin/activate && python scripts/ingest_path.py data/sample_docs --modality text
