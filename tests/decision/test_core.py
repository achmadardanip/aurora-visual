"""Module-3 core tests: verifier, fusion, conformal math, policies, aggregation."""

import json
import math
from pathlib import Path

import pytest
from aurora_decision.config import Settings
from aurora_decision.contract import AuroraBundle, Evidence
from aurora_decision.core.aggregation import aggregate_claim
from aurora_decision.core.calibration import (
    calibrate,
    conformal_quantile,
    prediction_set,
    safe_quantile_representation,
)
from aurora_decision.core.fuse import fuse, input_route
from aurora_decision.core.fusion import (
    build_features,
    evidence_only,
    rule_fusion,
    visual_only,
)
from aurora_decision.core.policies import apply_policy
from aurora_decision.core.reporting import csv_cell, generate_markdown, markdown_to_html
from aurora_decision.core.verifier import verify_atom_against_evidence


def load_fixture():
    return AuroraBundle.model_validate(
        json.loads(
            (
                Path(__file__).resolve().parents[2] / "aurora-decision" / "fixtures" / "fusion_demo.json"
            ).read_text()
        )
    )


def make_atom(
    atom_id="a000001",
    statement="Gedung Sate memiliki 6 lantai",
    role="quantity",
    negated=False,
    quantity=None,
    predicate="memiliki",
):
    from aurora_decision.contract import Atom

    return Atom.model_validate(
        {
            "atom_id": atom_id,
            "statement": statement,
            "role": role,
            "subject": None,
            "predicate": predicate,
            "object": None,
            "qualifiers": {"negated": negated, "quantity": quantity, "time": None, "location": None},
            "spans": [{"start": 0, "end": len(statement)}],
            "depends_on": [],
            "check_worthiness": 0.8,
            "parser_confidence": None,
        }
    )


def make_evidence(
    evidence_id="ev_" + "0" * 32 + "_000001",
    text="Gedung Sate memiliki 6 lantai.",
    group="igr-1",
    eligible=True,
):
    from aurora_decision.contract import sha

    return Evidence.model_validate(
        {
            "evidence_id": evidence_id,
            "atom_ids": [],
            "source": {
                "provider": "t",
                "kind": "news",
                "url": None,
                "title": "t",
                "publisher": "p",
                "language": "id",
                "published_at": None,
                "retrieved_at": "2026-09-15T00:00:00+00:00",
            },
            "content": {"text": text, "excerpt": text[:20], "sha256": sha(text), "status": "full"},
            "provenance": {
                "original_url": None,
                "archive_url": None,
                "first_seen_at": None,
                "captured_at": None,
                "date_basis": None,
                "discovery_method": "test",
                "duplicate_cluster_id": "dup-1",
                "independence_group_id": group,
                "temporal_eligible": eligible,
                "usage_note": None,
                "image_matches": [],
            },
            "relevance_score": 0.5,
            "credibility_score": 0.5,
            "rerank_score": 1.0,
            "score_breakdown": {},
        }
    )


class TestVerifier:
    def test_supports_matching_statement(self):
        atom = make_atom()
        evidence = make_evidence(
            text="Dokumen resmi menyatakan Gedung Sate memiliki 6 lantai sesuai catatan."
        )
        result = verify_atom_against_evidence(atom, evidence)
        assert result.stance == "Supports"
        assert result.quote and result.quote in evidence.content.text

    def test_contradicts_different_number(self):
        atom = make_atom(statement="Gedung Sate memiliki 6 lantai")
        evidence = make_evidence(
            text="Laporan resmi menyatakan Gedung Sate memiliki 3 lantai setelah renovasi."
        )
        result = verify_atom_against_evidence(atom, evidence)
        assert result.stance == "Contradicts"

    def test_contradicts_negation(self):
        atom = make_atom(statement="Gedung Sate dibuka untuk umum", predicate="dibuka")
        evidence = make_evidence(
            text="Gedung Sate tidak dibuka untuk umum sejak renovasi dimulai bulan lalu."
        )
        result = verify_atom_against_evidence(atom, evidence)
        assert result.stance == "Contradicts"

    def test_not_relevant_off_topic(self):
        atom = make_atom(statement="Gedung Sate memiliki 6 lantai")
        evidence = make_evidence(text="Timnas memenangkan pertandingan final dengan skor tiga banding satu.")
        assert verify_atom_against_evidence(atom, evidence).stance == "NotRelevant"

    def test_unavailable_content_is_not_relevant(self):
        evidence = make_evidence(text="")
        evidence.content.status = "unavailable"
        atom = make_atom()
        assert verify_atom_against_evidence(atom, evidence).stance == "NotRelevant"

    def test_attribution_report_is_not_support(self):
        atom = make_atom(statement="Gedung Sate memiliki 6 lantai")
        evidence = make_evidence(
            text="Sebuah unggahan mengklaim Gedung Sate memiliki 6 lantai; klaim ini belum diverifikasi."
        )
        result = verify_atom_against_evidence(atom, evidence)
        assert result.stance == "Unclear"


class TestFusion:
    def _link(self, atom_id, evidence_id, stance, group, eligible=True):
        from aurora_decision.contract import EvidenceLink

        return EvidenceLink.model_validate(
            {
                "evidence_id": evidence_id,
                "atom_id": atom_id,
                "stance": stance,
                "quote": None,
                "rationale": "test",
            }
        )

    def test_ten_copies_one_group_single_vote(self):
        atom = make_atom()
        links, evidence = [], []
        for i in range(10):
            eid = f"ev_{'0' * 32}_{i:06d}"
            evidence.append(make_evidence(evidence_id=eid, text="x", group="same-story"))
            links.append(self._link(atom.atom_id, eid, "Supports", "same-story"))
        features = build_features(atom, evidence, links, [])
        assert features.support_groups == 1  # not 10
        assert features.eligible_count == 10  # all visible

    def test_conflicting_groups_flagged(self):
        atom = make_atom()
        evidence = [
            make_evidence(evidence_id="ev_" + "0" * 32 + "_000001", group="g1"),
            make_evidence(evidence_id="ev_" + "0" * 32 + "_000002", group="g2"),
        ]
        links = [
            self._link(atom.atom_id, evidence[0].evidence_id, "Supports", "g1"),
            self._link(atom.atom_id, evidence[1].evidence_id, "Contradicts", "g2"),
        ]
        features = build_features(atom, evidence, links, [])
        assert features.conflicts

    def test_future_evidence_not_eligible(self):
        atom = make_atom()
        evidence = [make_evidence(eligible=False)]
        links = [self._link(atom.atom_id, evidence[0].evidence_id, "Supports", "g1")]
        features = build_features(atom, evidence, links, [])
        assert features.missing_evidence  # ineligible evidence doesn't count
        assert features.eligible_count == 0

    def test_route_baselines(self):
        from aurora_decision.core.fusion import FusionFeatures

        visual = FusionFeatures(atom_id="a", visual_status="Supported", missing_visual=False)
        assert visual_only(visual)[0] == "Supported"
        evidence_case = FusionFeatures(
            atom_id="a", support_groups=2, missing_evidence=False, eligible_count=2
        )
        assert evidence_only(evidence_case)[0] == "Supported"
        conflict = FusionFeatures(
            atom_id="a",
            support_groups=1,
            contradict_groups=1,
            conflicts=True,
            missing_evidence=False,
            eligible_count=2,
        )
        assert rule_fusion(conflict)[0] == "InsufficientEvidence"


class TestConformal:
    def test_quantile_tiny_n_math(self):
        # n=9, alpha=0.05: k=ceil(10*0.95)=10 > 9 -> infinity
        q, k, n = conformal_quantile([0.1] * 9, 0.05)
        assert k == 10 and n == 9 and math.isinf(q)
        # n=9, alpha=0.10: k=ceil(10*0.90)=9 <= 9 -> 9th order statistic
        q, k, n = conformal_quantile(sorted([0.5, 0.2, 0.9, 0.1, 0.3, 0.7, 0.4, 0.6, 0.8]), 0.10)
        assert k == 9 and q == 0.9

    def test_empty_group_infinite(self):
        q, k, n = conformal_quantile([], 0.1)
        assert math.isinf(q) and n == 0

    def test_infinity_sentinel_is_json_safe(self):
        sentinel = safe_quantile_representation(math.inf)
        assert sentinel == {"kind": "infinity", "value": None}
        json.dumps(sentinel)  # no NaN/Infinity leak

    def test_prediction_set_membership(self):
        artifact = calibrate(
            [{"id": f"s{i}", "score": 0.2, "group": ("quantity", "Supported")} for i in range(25)],
            alpha=0.1,
            input_route="fused",
            feature_schema_sha="x",
        )
        members, diagnostics = prediction_set(
            {"Supported": 0.95, "Contradicted": 0.03, "InsufficientEvidence": 0.02}, "quantity", artifact
        )
        assert "Supported" in members
        assert "quantity|Contradicted" not in artifact.groups  # absent group
        assert diagnostics["Contradicted"]["fallback"] is True or diagnostics["Contradicted"]["n"] == 0

    def test_route_mismatch_detected(self):
        from aurora_decision.core.calibration import matches_route

        artifact = calibrate(
            [{"id": "s", "score": 0.2, "group": ("time", "Supported")}],
            alpha=0.1,
            input_route="evidence_only",
            feature_schema_sha="sha1",
        )
        ok, reason = matches_route(
            artifact, "fused", "sha1", ["Supported", "Contradicted", "InsufficientEvidence"]
        )
        assert not ok and "input_route" in reason


class TestPolicies:
    def test_uncalibrated_abstains(self):
        status, abstain, reasons = apply_policy("Supported", None, {"calibrated": False})
        assert status == "InsufficientEvidence" and abstain and "UNCALIBRATED" in reasons

    def test_empty_and_ambiguous_sets_abstain(self):
        _, abstain1, r1 = apply_policy("Supported", [], {"calibrated": True})
        assert abstain1 and "EMPTY_PREDICTION_SET" in r1
        _, abstain2, r2 = apply_policy("Supported", ["Supported", "Contradicted"], {"calibrated": True})
        assert abstain2 and "AMBIGUOUS_PREDICTION_SET" in r2

    def test_singleton_supported_decides(self):
        status, abstain, reasons = apply_policy(
            "Supported", ["Supported"], {"calibrated": True, "evidence_backed": True}
        )
        assert status == "Supported" and not abstain and not reasons

    def test_singleton_ie_still_abstains(self):
        status, abstain, reasons = apply_policy(
            "InsufficientEvidence", ["InsufficientEvidence"], {"calibrated": True}
        )
        assert status == "InsufficientEvidence" and abstain and "INSUFFICIENT_EVIDENCE" in reasons


class TestAggregation:
    def test_one_essential_contradiction_contradicts_claim(self):
        atoms = [
            make_atom("a000001", "dibangun 1920", "time"),
            make_atom("a000002", "punya 6 lantai", "quantity"),
        ]
        decisions = {
            "a000001": {"base_label": "Contradicted", "evidence_backed": True, "conflicts": False},
            "a000002": {"base_label": "Supported", "evidence_backed": True, "conflicts": False},
        }
        label, _, _, details = aggregate_claim(atoms, decisions, {})
        assert label == "Contradicted"

    def test_supported_requires_all_essential_resolved(self):
        atoms = [
            make_atom("a000001", "dibangun 1920", "time"),
            make_atom("a000002", "punya 6 lantai", "quantity"),
        ]
        decisions = {
            "a000001": {"base_label": "Supported", "evidence_backed": True, "conflicts": False},
            "a000002": {"base_label": "InsufficientEvidence", "evidence_backed": False, "conflicts": False},
        }
        label, rationale, coverage, _ = aggregate_claim(atoms, decisions, {})
        assert label == "InsufficientEvidence"
        assert "belum terselesaikan" in rationale
        assert coverage < 1.0


class TestFusePipeline:
    def test_demo_fixture_end_to_end(self, tmp_path):
        settings = Settings(data_dir=tmp_path)
        settings.prepare()
        out = fuse(load_fixture(), settings, lambda _: None)
        decision = out.decision
        # Year atom contradicted by official source; floor count supported.
        base = {v.atom_id: v.base_label for v in decision.atomic_verdicts}
        assert base["a000001"] == "Contradicted"
        assert base["a000002"] == "Supported"
        # Live/demo without matching claim calibration: operational abstain.
        assert decision.final_verdict == "InsufficientEvidence"
        assert "UNCALIBRATED" in decision.abstention_reasons
        # Duplicate syndication: floor atom supported but coverage counts groups.
        assert out.extensions["aurora_decision"]["coverage"] < 1.0
        # Report exists and is grounded.
        report = out.extensions["aurora_decision"]["report_markdown"]
        assert "Laporan Keputusan AURORA" in report and "Gedung Sate" in report

    def test_missing_analysis_raises(self, tmp_path):
        settings = Settings(data_dir=tmp_path)
        settings.prepare()
        bundle = load_fixture()
        data = json.loads(bundle.model_dump_json())
        data["analysis"] = None
        data["retrieval"] = None
        with pytest.raises(ValueError, match="INPUT_ANALYSIS_REQUIRED"):
            fuse(AuroraBundle.model_validate(data), settings)

    def test_input_routes(self):
        bundle = load_fixture()
        route, warnings = input_route(bundle)
        assert route == "evidence_only"  # fixture has no visual assessments
        assert warnings
        data = json.loads(bundle.model_dump_json())
        data["retrieval"] = None
        route2, _ = input_route(AuroraBundle.model_validate(data))
        assert route2 == "visual_only"

    def test_fake_quote_rejected(self, tmp_path):
        """A link whose quote is not an exact substring must not survive fusion."""
        settings = Settings(data_dir=tmp_path)
        settings.prepare()
        bundle = load_fixture()
        data = json.loads(bundle.model_dump_json())
        from aurora_decision.contract import sha

        replacement = "Teks yang sama sekali berbeda tanpa angka lantai atau tahun."
        for item in data["retrieval"]["evidence_list"]:
            item["content"]["text"] = replacement
            item["content"]["excerpt"] = replacement[:20]
            item["content"]["sha256"] = sha(replacement)
        validated = AuroraBundle.model_validate(data)
        out = fuse(validated, settings, lambda _: None)
        for verdict in out.decision.atomic_verdicts:
            for link in verdict.evidence_links:
                if link.quote is not None:
                    item = next(e for e in out.retrieval.evidence_list if e.evidence_id == link.evidence_id)
                    assert link.quote in item.content.text


class TestReporting:
    def test_markdown_grounded_in_bundle(self):
        settings = Settings()
        settings.prepare()
        out = fuse(load_fixture(), settings, lambda _: None)
        report = generate_markdown(out)
        for evidence_id in out.decision.decision_report.evidence_ids:
            assert evidence_id in report
        # Any AI-indication mention must carry its non-verdict disclaimer.
        if "Indikasi konten AI" in report:
            assert "tidak memengaruhi putusan faktual" in report or "bukan putusan" in report

    def test_html_escapes_untrusted(self):
        html = markdown_to_html("# T\n\n- **x** <script>alert(1)</script>")
        assert "<script>" not in html and "&lt;script&gt;" in html

    def test_csv_formula_escaping(self):
        assert csv_cell("=SUM(A1)").startswith("'")
        assert csv_cell(None) == ""
