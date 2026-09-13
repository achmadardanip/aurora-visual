import json
from pathlib import Path
from uuid import UUID

from app.api.main import app
from app.models.contract import Analysis, AuroraBundle, RunInfo, atom_set_id, canonical, sha
from aurora_visual.atomization.parser import RuleAtomizer

root = Path(__file__).resolve().parents[1]
(root / "contracts/aurora.schema.json").write_text(
    json.dumps(AuroraBundle.model_json_schema(), indent=2) + "\n"
)
(root / "contracts/openapi.json").write_text(json.dumps(app.openapi(), indent=2) + "\n")
caption = "📷 Dua mahasiswa tidak membawa 2,5 kg barang di Monas pada September 2026"
b = AuroraBundle(
    schema_version="1.0.0",
    case_id=str(UUID(int=1)),
    claim_revision=1,
    mode="demo",
    created_at="2026-09-12T00:00:00Z",
    input={"claim_text": caption, "language": "id", "images": [], "as_of": None},
    analysis=None,
    retrieval=None,
    decision=None,
    warnings=[],
    extensions={"golden": {"fraction": 0.125, "non_ascii": "é漢📷", "unknown": None}},
)
atoms = RuleAtomizer().parse(caption).atoms
run = RunInfo(
    run_id=str(UUID(int=2)),
    mode="demo",
    started_at="2026-09-12T00:00:00Z",
    finished_at="2026-09-12T00:00:00Z",
    status="partial",
    versions={"importer": "golden-fixture-v1"},
    warnings=[],
)
b.analysis = Analysis(
    run=run, atom_set_id=atom_set_id(b, atoms), atomic_claims=atoms, visual_assessments=[], ocr=[]
)
b = AuroraBundle.model_validate(b.model_dump(mode="json"))
(root / "contracts/golden-bundle.json").write_text(b.model_dump_json(indent=2) + "\n")
raw = b.model_dump(mode="json")
(root / "contracts/jcs-golden.json").write_text(
    json.dumps(
        {"input": raw, "canonical_utf8": canonical(raw).decode(), "sha256": sha(canonical(raw))},
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
)
print("Pydantic, JSON Schema, OpenAPI and Unicode JCS golden vector exported")
