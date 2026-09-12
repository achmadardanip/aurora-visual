import copy
from pathlib import Path

import pytest
from app.models.contract import AuroraBundle, strict_json
from pydantic import ValidationError


def base():
    return strict_json(Path("contoh/02_INPUT_FUSION_DEMO.json").read_bytes())


@pytest.mark.parametrize("mutation", ["excerpt", "hash", "reference", "atom_set", "run_mode"])
def test_evidence_invariants(mutation):
    b = base()
    e = b["retrieval"]["evidence_list"][0]
    if mutation == "excerpt":
        e["content"]["excerpt"] = "fabricated quotation"
    elif mutation == "hash":
        e["content"]["sha256"] = "0" * 64
    elif mutation == "reference":
        e["atom_ids"] = ["a900000"]
    elif mutation == "atom_set":
        b["retrieval"]["atom_set_id"] = "aset_" + "0" * 64
    elif mutation == "run_mode":
        b["retrieval"]["run"]["mode"] = "live"
    with pytest.raises(ValidationError):
        AuroraBundle.model_validate(b)


def decision_bundle():
    b = base()
    calibration = {
        "status": "uncalibrated",
        "calibration_id": None,
        "method": None,
        "alpha": None,
        "sample_count": 0,
        "group": None,
        "fallback_used": None,
        "validity_notes": ["No calibration artifact"],
    }
    atomic = [
        {
            "atom_id": a["atom_id"],
            "base_label": "Supported",
            "status": "InsufficientEvidence",
            "probabilities": None,
            "confidence_set": None,
            "abstention_flag": True,
            "abstention_reasons": ["uncalibrated"],
            "evidence_links": [],
            "visual_atom_ids": [a["atom_id"]],
            "explanation": "Test only",
            "calibration": copy.deepcopy(calibration),
        }
        for a in b["analysis"]["atomic_claims"]
    ]
    b["decision"] = {
        "run": copy.deepcopy(b["analysis"]["run"]),
        "atom_set_id": b["analysis"]["atom_set_id"],
        "atomic_verdicts": atomic,
        "base_label": "Supported",
        "final_verdict": "InsufficientEvidence",
        "probabilities": None,
        "confidence_set": None,
        "abstention_flag": True,
        "abstention_reasons": ["uncalibrated"],
        "calibration": calibration,
        "decision_report": {
            "summary": "Test",
            "key_findings": [],
            "unresolved_questions": [],
            "evidence_ids": [],
            "forensic_signal_ids": [],
            "limitations": [],
            "suggested_next_steps": [],
            "misinformation_category": None,
        },
        "human_review": None,
    }
    b["decision"]["run"]["status"] = "completed"
    return b


def test_decision_null_calibration_abstains():
    assert AuroraBundle.model_validate(decision_bundle()).decision.abstention_flag


@pytest.mark.parametrize(
    "mutation", ["operational_label", "false_set", "duplicate_set", "missing_atom", "dangling_quote"]
)
def test_decision_guards(mutation):
    b = decision_bundle()
    d = b["decision"]
    if mutation == "operational_label":
        d["final_verdict"] = "Supported"
    elif mutation == "false_set":
        d["confidence_set"] = []
    elif mutation == "duplicate_set":
        d["confidence_set"] = ["Supported", "Supported"]
    elif mutation == "missing_atom":
        d["atomic_verdicts"].pop()
    elif mutation == "dangling_quote":
        d["atomic_verdicts"][0]["evidence_links"] = [
            {
                "evidence_id": b["retrieval"]["evidence_list"][0]["evidence_id"],
                "atom_id": d["atomic_verdicts"][0]["atom_id"],
                "stance": "Supports",
                "quote": "fabricated quote",
                "rationale": "test",
            }
        ]
    with pytest.raises(ValidationError):
        AuroraBundle.model_validate(b)
