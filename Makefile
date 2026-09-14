.PHONY: setup dev demo test lint schema smoke browser-test evaluate train experiments build backup verify-backup restore-backup mafindo-live-test deepseek-live-test tune-smoke modal-smoke mafindo-corpus
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
	uv run python scripts/normalize_pytest_report.py

smoke:
	uv run python scripts/smoke.py

browser-test:
	npm exec --prefix frontend -- playwright install chromium --only-shell
	uv run python scripts/browser_smoke.py

train:
	uv run aurora train --smoke --epochs 3 --output artifacts/checkpoints/smoke.pt

evaluate: train
	uv run aurora evaluate --smoke --checkpoint artifacts/checkpoints/smoke.pt --output artifacts/reports/evaluation-smoke.json

tune-smoke:
	uv run aurora tune --smoke --trials 15 --epochs 3 --output artifacts/reports/tune-smoke.json --best-config artifacts/tuning/best-config-smoke.json

modal-smoke:
	uv run modal run scripts/modal_train.py --smoke --epochs 3 --tune --trials 3 --gpu T4 --output artifacts/reports/modal-run.json

mafindo-corpus:
	uv run aurora mafindo-corpus --limit 100000 --output data/mafindo/corpus.jsonl --summary artifacts/reports/mafindo-corpus.json

experiments:
	uv run aurora experiments --smoke --epochs 2 --output artifacts/reports/ablations-smoke.json

build:
	npm run build --prefix frontend

backup:
	uv run python scripts/backup.py --data-dir "$${AURORA_DATA_DIR:-var}" --output-dir "$${AURORA_BACKUP_DIR:-backups}"

verify-backup:
	@test -n "$(ARCHIVE)" || (echo "Usage: make verify-backup ARCHIVE=backups/aurora-backup-....tar.gz" >&2; exit 2)
	uv run python scripts/verify_backup.py "$(ARCHIVE)"

restore-backup:
	@test -n "$(ARCHIVE)" || (echo "Usage: make restore-backup ARCHIVE=backups/aurora-backup-....tar.gz RESTORE_DIR=/empty/path" >&2; exit 2)
	@test -n "$(RESTORE_DIR)" || (echo "Usage: make restore-backup ARCHIVE=backups/aurora-backup-....tar.gz RESTORE_DIR=/empty/path" >&2; exit 2)
	uv run python scripts/restore_backup.py "$(ARCHIVE)" --data-dir "$(RESTORE_DIR)"

mafindo-live-test:
	uv run python scripts/mafindo_live_test.py

deepseek-live-test:
	uv run aurora deepseek-test --output artifacts/reports/deepseek-test.json
