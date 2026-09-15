# AURORA — Three-Module Misinformation Verification System

AURORA is a research system that checks whether an image caption's claims are
supported by what is actually visible in the image and by external evidence.

**Repository layout (one system, three coordinated apps):**

| Module | App | API | UI | What it does |
| --- | --- | --- | --- | --- |
| 1 | `backend/` + `frontend/` (aurora-visual) | :8101 | http://127.0.0.1:5171 | Splits the caption into atomic claims; checks each against the image (OCR, regions, visual entailment → Supported / Contradicted / Unobservable). |
| 2 | `aurora-evidence/` | :8102 | http://127.0.0.1:5172 | Searches local fact-check corpus (BM25) + optional web providers; dedups syndication; runs AI-generated-content detectors on image and text. |
| 3 | `aurora-decision/` | :8103 | http://127.0.0.1:5173 | Verifies each atom against each evidence document, fuses multi-source evidence with independence groups, applies conformal abstention, generates an auditable report, records human review. Orchestrates modules 1→2→3. |

All three apps share the same `AuroraBundle` v1.0.0 data contract
(RFC 8785 canonical JSON, content-addressed media, deterministic atom-set
hashes) and can also run standalone.

## Quick start (all three apps, one command)

```sh
make setup     # first time only: dependencies, migrations, frontend installs
make system    # starts API :8101-8103 and UI :5171-5173
```

Then open **http://127.0.0.1:5173** (the Decision app) and use
**“Pipeline penuh”** to run a caption + image through all three modules.

Individual apps (optional):

```sh
make dev                          # module 1 only
uv run uvicorn aurora_evidence.api.main:app --port 8102 --cwd aurora-evidence/backend
uv run uvicorn aurora_decision.api.main:app --port 8103 --cwd aurora-decision/backend
```

## What works out of the box (no API keys)

- Module 1: rules-based claim atomizer, local color baseline, OpenCLIP
  (weights downloaded), OCR (Tesseract ind+eng), C2PA provenance validation.
- Module 2: real BM25 search over the labeled synthetic demo corpus
  (`aurora-evidence/fixtures/demo_corpus.jsonl`), URL/text/shingle dedup,
  dHash local image index, deterministic query planning.
- Module 3: deterministic atom–evidence verifier, rule fusion with
  independence groups, global + Mondrian split conformal (demo artifacts),
  abstention policy, Markdown/HTML/CSV report generator, human review,
  upstream orchestration across all three services.
- Demo modes everywhere use clearly-labeled synthetic fixtures.

## Optional providers (env or `.env.evidence` in aurora-evidence/)

| Variable | Enables | Where |
| --- | --- | --- |
| `TAVILY_API_KEY` | Web search (module 2) | aurora-evidence |
| `GPTZERO_API_KEY` | AI-text detection (module 2) | aurora-evidence |
| `HIVE_API_KEY` | AI-image detection (module 2) | aurora-evidence |
| `AURORA_DEEPSEEK_*` | DeepSeek multimodal analysis (module 1) | backend/.env |
| `AURORA_HIVE_*` | Hive stage-1 detection + VLM (module 1) | backend/.env |

## Tests

```sh
make test        # 367 tests: unit, contract, integration, e2e browser smoke
```

## Honest limitations

- The deterministic verifier only understands narrow explicit relations
  (entity overlap, adjacent negation, number/year conflicts). Everything else
  is `Unclear`, never a guess.
- Live claim-level decisions always abstain without a matching calibration
  artifact (atom-level sets exist in demo mode). This is by design: the
  system refuses to present uncalibrated scores as decisions.
- AI-detector scores are provider indications of generated content, never
  evidence that a claim is false.
- The demo corpus is synthetic; live research evaluation requires licensed
  datasets (see docs in each app).
