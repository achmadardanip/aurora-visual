"""Generate the labeled synthetic fusion demo fixture with correct contract hashes."""

import json
import sys
from pathlib import Path
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aurora-decision" / "backend"))

from aurora_decision.contract import Analysis, Atom, AuroraBundle, RunInfo, atom_set_id, sha

CASE_ID = "11111111-2222-3333-4444-555555555555"
RUN_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
RETRIEVAL_RUN = "ffffffff-1111-2222-3333-444444444444"
CLAIM = "Gedung sate dibangun pada tahun 1920 dan memiliki 6 lantai."

CAPTION_ATOMS = [
    {
        "atom_id": "a000001",
        "statement": "Gedung Sate dibangun pada tahun 1920",
        "role": "time",
        "subject": "Gedung Sate",
        "predicate": "dibangun",
        "object": None,
        "qualifiers": {"negated": False, "quantity": None, "time": "1920", "location": None},
        "spans": [{"start": 0, "end": 36}],
        "depends_on": [],
        "check_worthiness": 0.9,
        "parser_confidence": None,
    },
    {
        "atom_id": "a000002",
        "statement": "Gedung Sate memiliki 6 lantai",
        "role": "quantity",
        "subject": "Gedung Sate",
        "predicate": "memiliki",
        "object": "lantai",
        "qualifiers": {"negated": False, "quantity": 6, "time": None, "location": None},
        "spans": [{"start": 41, "end": 59}],
        "depends_on": [],
        "check_worthiness": 0.8,
        "parser_confidence": None,
    },
]

# Evidence: one authoritative doc contradicting the year, one fact-check supporting
# the floor count, plus a syndicated duplicate of the second (same independence
# group). Clearly labeled synthetic.
DOCS = [
    {
        "seq": 1,
        "kind": "official",
        "title": "[DEMO SINTETIS] Situs resmi: sejarah Gedung Sate",
        "text": "Situs resmi pemprov demo mencatat Gedung Sate dibangun pada tahun 1921 dan dirancang arsitek Belanda. Bangunan ini memiliki 6 lantai. Dokumen sintetis untuk pengujian fusi.",
        "publisher": "Pemprov Demo",
        "published_at": "2020-05-10T08:00:00+00:00",
        "group": None,
        "atom_ids": ["a000001", "a000002"],
    },
    {
        "seq": 2,
        "kind": "fact_check",
        "title": "[DEMO SINTETIS] Cek fakta: jumlah lantai Gedung Sate",
        "text": "Tim cek fakta demo memeriksa klaim jumlah lantai Gedung Sate. Berdasarkan dokumen resmi, Gedung Sate memiliki 6 lantai. Dokumen sintetis untuk pengujian.",
        "publisher": "Cek Fakta Demo",
        "published_at": "2026-01-15T09:00:00+00:00",
        "group": "igr-demo-lantai",
        "atom_ids": ["a000002"],
    },
    {
        "seq": 3,
        "kind": "news",
        "title": "[DEMO SINTETIS] Portal menyiarkan ulang cek fakta lantai",
        "text": "Portal demo menyiarkan ulang pemeriksaan: Berdasarkan dokumen resmi, Gedung Sate memiliki 6 lantai. Salinan sindikasi dari Cek Fakta Demo. Sintetis untuk pengujian.",
        "publisher": "Portal Demo",
        "published_at": "2026-01-15T11:00:00+00:00",
        "group": "igr-demo-lantai",
        "atom_ids": ["a000002"],
    },
]


def build() -> dict:
    bundle_data = {
        "schema_version": "1.0.0",
        "case_id": CASE_ID,
        "claim_revision": 1,
        "mode": "demo",
        "created_at": "2026-09-15T00:00:00+00:00",
        "input": {"claim_text": CLAIM, "language": "id", "images": [], "as_of": None},
        "analysis": None,
        "retrieval": None,
        "decision": None,
        "warnings": [],
        "extensions": {},
    }
    atoms = [Atom.model_validate(a) for a in CAPTION_ATOMS]
    temp = AuroraBundle.model_validate(bundle_data)
    aset = atom_set_id(temp, atoms)
    analysis = {
        "run": {
            "run_id": RUN_ID,
            "mode": "demo",
            "started_at": "2026-09-15T00:00:00+00:00",
            "finished_at": "2026-09-15T00:00:01+00:00",
            "status": "completed",
            "versions": {"service": "demo-importer", "parser": "manual-annotation-v1"},
            "warnings": [
                {
                    "code": "SYNTHETIC_FIXTURE",
                    "message": "Anotasi atom manual sintetis untuk pengujian fusi.",
                    "component": "demo",
                }
            ],
        },
        "atom_set_id": aset,
        "atomic_claims": CAPTION_ATOMS,
        "visual_assessments": [],
        "ocr": [],
    }
    run_hex = RETRIEVAL_RUN.replace("-", "")
    evidence = []
    for doc in DOCS:
        text = doc["text"]
        excerpt = next(
            (s.strip() for s in text.split(". ") if "lantai" in s or "1921" in s),
            text[:80],
        )
        evidence.append(
            {
                "evidence_id": f"ev_{run_hex}_{doc['seq']:06d}",
                "atom_ids": doc["atom_ids"],
                "source": {
                    "provider": "demo-fixtures",
                    "kind": doc["kind"],
                    "url": f"https://demo.example.org/fusion/{doc['seq']}",
                    "title": doc["title"],
                    "publisher": doc["publisher"],
                    "language": "id",
                    "published_at": doc["published_at"],
                    "retrieved_at": "2026-09-15T00:00:02+00:00",
                },
                "content": {
                    "text": text,
                    "excerpt": excerpt,
                    "sha256": sha(text),
                    "status": "full",
                },
                "provenance": {
                    "original_url": f"https://demo.example.org/fusion/{doc['seq']}",
                    "archive_url": None,
                    "first_seen_at": None,
                    "captured_at": None,
                    "date_basis": "published_at",
                    "discovery_method": "demo_fixture",
                    "duplicate_cluster_id": f"dup_{sha(text)[:24]}",
                    "independence_group_id": doc["group"] or f"igr_{sha(doc['title'])[:20]}",
                    "temporal_eligible": True,
                    "usage_note": None,
                    "image_matches": [],
                },
                "relevance_score": 0.9,
                "credibility_score": 0.8 if doc["kind"] in ("official", "fact_check") else 0.5,
                "rerank_score": 1.0,
                "score_breakdown": {"demo": 1.0},
            }
        )
    retrieval = {
        "run": {
            "run_id": RETRIEVAL_RUN,
            "mode": "demo",
            "started_at": "2026-09-15T00:00:02+00:00",
            "finished_at": "2026-09-15T00:00:03+00:00",
            "status": "completed",
            "versions": {"service": "demo-fixtures"},
            "warnings": [],
        },
        "atom_set_id": aset,
        "evidence_list": evidence,
        "forensic_signals": [],
        "query_log": [],
        "provider_status": [
            {"provider": "demo-fixtures", "capability": "fact_check_search", "status": "ok", "message": None}
        ],
    }
    bundle_data["analysis"] = analysis
    bundle_data["retrieval"] = retrieval
    bundle_data["extensions"]["aurora_evidence"] = {"fixture": "fusion-demo"}
    # Final validation through the contract model.
    validated = AuroraBundle.model_validate(bundle_data)
    return json.loads(validated.model_dump_json())


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "fusion_demo.json"
    out.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print("fixture written:", out)
