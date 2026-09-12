.PHONY: setup dev demo test lint schema smoke browser-test evaluate train experiments build mafindo-live-test
setup:
	uv sync --frozen --python 3.12 --extra ml --extra dev
	npm ci --prefix frontend
	uv run alembic upgrade head
	uv run python scripts/export_contract.py

dev:
	uv run python scripts/dev.py

demo: dev

lint:
	uv run ruff check backend tests scripts
	uv run ruff format --check backend tests scripts
	npm run typecheck --prefix frontend
	npm exec --prefix frontend -- prettier --check frontend/src frontend/vite.config.ts frontend/tsconfig.json

schema:
	uv run python scripts/export_contract.py
	node tests/contract/jcs.mjs

test: lint schema
	uv run pytest -q --junitxml=artifacts/reports/pytest.xml

smoke:
	uv run python scripts/smoke.py

browser-test:
	npm exec --prefix frontend -- playwright install chromium --only-shell
	uv run python scripts/browser_smoke.py

train:
	uv run aurora train --smoke --epochs 3 --output artifacts/checkpoints/smoke.pt

evaluate: train
	uv run aurora evaluate --smoke --checkpoint artifacts/checkpoints/smoke.pt --output artifacts/reports/evaluation-smoke.json

experiments:
	uv run aurora experiments --smoke --epochs 2 --output artifacts/reports/ablations-smoke.json

build:
	npm run build --prefix frontend

mafindo-live-test:
	uv run python scripts/mafindo_live_test.py
