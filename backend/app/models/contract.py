"""AURORA 1.0.0 wire contract. Nullability never implies an omitted field."""

import hashlib
import json
import math
from datetime import timezone
from typing import Annotated, Any, Literal
from uuid import UUID

import rfc8785
from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Score = Annotated[float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Bool = Annotated[bool, Field(strict=True)]
NonnegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(ge=1, strict=True)]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Time = Annotated[AwareDatetime, AfterValidator(lambda x: x.astimezone(timezone.utc))]
VisualLabel = Literal["Supported", "Contradicted", "Unobservable"]
FactLabel = Literal["Supported", "Contradicted", "InsufficientEvidence"]
Role = Literal["actor", "action", "object", "attribute", "location", "time", "quantity", "relation", "cause"]
Mode = Literal["demo", "live"]


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def canonical(data: Any) -> bytes:
    return rfc8785.dumps(data)


def strict_json(data: str | bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    result = json.loads(
        data, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
    )
    canonical(result)  # Also rejects unsafe integers, overflow and non-finite numbers.
    return result


def bbox_check(v):
    if not (0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1):
        raise ValueError("Invalid normalized bbox")
    return v


BBox = Annotated[tuple[Number, Number, Number, Number], AfterValidator(bbox_check)]


def probabilities(v):
    if v is not None and abs(sum(v.values()) - 1) > 1e-6:
        raise ValueError("Probabilities must sum to one")
    return v


VisualProb = Annotated[dict[VisualLabel, Score], AfterValidator(probabilities)]
FactProb = Annotated[dict[FactLabel, Score], AfterValidator(probabilities)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_distributions(self):
        for name in ("probabilities",):
            if hasattr(self, name) and getattr(self, name) is not None:
                expected = (
                    {"Supported", "Contradicted", "Unobservable"}
                    if type(self).__name__ == "VisualAssessment"
                    else {"Supported", "Contradicted", "InsufficientEvidence"}
                )
                if set(getattr(self, name)) != expected:
                    raise ValueError("Incomplete label distribution")
        return self


class Warning(Model):
    code: str
    message: str
    component: str


class MediaRef(Model):
    asset_id: str
    sha256: Hash
    media_type: Annotated[str, Field(min_length=1)]
    width: PositiveInt
    height: PositiveInt
    uri: str

    @model_validator(mode="after")
    def identity(self):
        if self.asset_id != "asset_" + self.sha256:
            raise ValueError("Asset identity mismatch")
        if (
            (self.uri.startswith("/") and not self.uri.startswith("/api/"))
            or self.uri.startswith(("file:", "\\"))
            or ".." in self.uri.split("/")
        ):
            raise ValueError("Local or unsafe URI is forbidden; use relative API/ZIP URI or HTTPS")
        return self


class RunInfo(Model):
    run_id: str
    mode: Mode
    started_at: Time
    finished_at: Time
    status: Literal["completed", "partial", "failed"]
    versions: dict[str, str]
    warnings: list[Warning]

    @model_validator(mode="after")
    def identity(self):
        if str(UUID(self.run_id)) != self.run_id or self.finished_at < self.started_at:
            raise ValueError("Invalid run identity or timestamps")
        return self


class Qualifiers(Model):
    negated: Bool
    quantity: Number | None
    time: str | None
    location: str | None


class Span(Model):
    start: NonnegativeInt
    end: NonnegativeInt

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Empty/reversed span")
        return self


class Atom(Model):
    atom_id: Annotated[str, Field(pattern=r"^a[0-9]{6}$")]
    statement: Annotated[str, Field(min_length=1, max_length=10000)]
    role: Role
    subject: str | None
    predicate: Annotated[str, Field(min_length=1)]
    object: str | None
    qualifiers: Qualifiers
    spans: list[Span]
    depends_on: list[str]
    check_worthiness: Score
    parser_confidence: Score | None


class Region(Model):
    region_id: Annotated[str, Field(pattern=r"^rg_[0-9a-f]{32}_[0-9]{6}$")]
    asset_id: str
    bbox: BBox
    score: Score | None
    description: str


class VisualAssessment(Model):
    atom_id: str
    visual_status: VisualLabel
    probabilities: VisualProb | None
    unmatched_mass: Score | None
    observability_score: Score | None
    supporting_regions: list[Region]
    contradicting_regions: list[Region]
    counter_evidence: str | None
    rationale: str
    inference_kind: Literal["trained", "heuristic", "fixture", "unavailable"]

    @model_validator(mode="after")
    def contradiction(self):
        if self.visual_status == "Contradicted" and (
            not self.contradicting_regions or not self.counter_evidence
        ):
            raise ValueError("Contradiction requires explicit region and counter_evidence")
        if self.inference_kind in ("heuristic", "unavailable") and self.probabilities is not None:
            raise ValueError("No distribution available for heuristic/unavailable inference")
        return self


class OCRItem(Model):
    text: str
    bbox: BBox
    confidence: Score | None
    language: str | None


class Analysis(Model):
    run: RunInfo
    atom_set_id: str
    atomic_claims: list[Atom]
    visual_assessments: list[VisualAssessment]
    ocr: list[OCRItem]


class Source(Model):
    provider: str
    kind: Literal[
        "fact_check", "news", "official", "web", "image_provenance", "local_corpus", "user_supplied"
    ]
    url: str | None
    title: str
    publisher: str | None
    language: str | None
    published_at: Time | None
    retrieved_at: Time


class Content(Model):
    text: str
    excerpt: str
    sha256: Hash
    status: Literal["full", "excerpt_only", "snippet_only", "unavailable"]

    @model_validator(mode="after")
    def integrity(self):
        if self.sha256 != sha(self.text) or self.excerpt not in self.text:
            raise ValueError("Evidence hash or excerpt mismatch")
        return self


class ImageMatch(Model):
    asset_id: str
    matched_url: str | None
    match_type: Literal["exact", "near_duplicate", "semantic"]
    score: Score | None


class Provenance(Model):
    original_url: str | None
    archive_url: str | None
    first_seen_at: Time | None
    captured_at: Time | None
    date_basis: str | None
    discovery_method: str
    duplicate_cluster_id: Annotated[str, Field(min_length=1)]
    independence_group_id: Annotated[str, Field(min_length=1)]
    temporal_eligible: Bool | None
    usage_note: str | None
    image_matches: list[ImageMatch]


class Evidence(Model):
    evidence_id: Annotated[str, Field(pattern=r"^ev_[0-9a-f]{32}_[0-9]{6}$")]
    atom_ids: list[str]
    source: Source
    content: Content
    provenance: Provenance
    relevance_score: Score | None
    credibility_score: Score | None
    rerank_score: Number | None
    score_breakdown: dict[str, Number]


class Target(Model):
    kind: Literal["claim_text", "image", "ocr_text"]
    asset_id: str | None
    text_sha256: Hash | None


class RawScale(Model):
    min: Number
    max: Number
    higher_means_ai: Bool


class ForensicSignal(Model):
    signal_id: Annotated[str, Field(pattern=r"^fs_[0-9a-f]{32}_[0-9]{6}$")]
    target: Target
    modality: Literal["image", "text"]
    provider: str
    model_version: str | None
    task: Literal["ai_generation_detection"]
    status: Literal["ok", "inconclusive", "unsupported", "unavailable", "failed"]
    raw_score: Number | None
    raw_scale: RawScale | None
    ai_generated_score: Score | None
    raw_label: str | None
    assessment: Literal["likely_ai_generated", "likely_human_or_camera", "uncertain", "not_assessed"]
    calibration_status: Literal["unknown", "provider_claimed", "locally_validated", "not_applicable"]
    applicable_language: str | None
    limitations: list[str]
    analyzed_at: Time
    error_code: str | None

    @model_validator(mode="after")
    def consistent(self):
        if self.status in ("unsupported", "unavailable", "failed") and (
            self.assessment != "not_assessed" or self.ai_generated_score is not None
        ):
            raise ValueError("Unavailable forensic signal is not an assessment")
        if self.status == "inconclusive" and self.assessment != "uncertain":
            raise ValueError("Inconclusive signal must be uncertain")
        if self.status != "ok" and self.assessment == "likely_ai_generated":
            raise ValueError("Non-ok forensic accusation")
        if self.raw_scale:
            lo, hi = self.raw_scale.min, self.raw_scale.max
            if lo >= hi or (self.raw_score is not None and not lo <= self.raw_score <= hi):
                raise ValueError("Invalid raw scale/score")
        if self.ai_generated_score is not None:
            if self.raw_score is None or self.raw_scale is None:
                raise ValueError("Unknown detector scale")
            score = (self.raw_score - self.raw_scale.min) / (self.raw_scale.max - self.raw_scale.min)
            if not self.raw_scale.higher_means_ai:
                score = 1 - score
            if abs(score - self.ai_generated_score) > 1e-6:
                raise ValueError("Detector normalization mismatch")
        if self.target.kind == "image":
            if (
                self.modality != "image"
                or self.target.asset_id is None
                or self.target.text_sha256 is not None
            ):
                raise ValueError("Invalid image target")
        elif self.modality != "text" or self.target.asset_id is not None or self.target.text_sha256 is None:
            raise ValueError("Invalid text target")
        return self


class QueryLog(Model):
    query_id: str
    atom_ids: list[str]
    provider: str
    query: str
    duration_ms: Annotated[Number, Field(ge=0)]
    result_count: NonnegativeInt


class ProviderStatus(Model):
    provider: str
    capability: str
    status: Literal["ok", "disabled", "unconfigured", "rate_limited", "failed", "unsupported"]
    message: str | None


class Retrieval(Model):
    run: RunInfo
    atom_set_id: str | None
    evidence_list: list[Evidence]
    forensic_signals: list[ForensicSignal]
    query_log: list[QueryLog]
    provider_status: list[ProviderStatus]


class EvidenceLink(Model):
    evidence_id: str
    atom_id: str
    stance: Literal["Supports", "Contradicts", "NotRelevant", "Unclear"]
    quote: str | None
    rationale: str


class Calibration(Model):
    status: Literal["calibrated", "uncalibrated", "demo_only"]
    calibration_id: str | None
    method: str | None
    alpha: Number | None
    sample_count: NonnegativeInt
    group: str | None
    fallback_used: str | None
    validity_notes: list[str]

    @model_validator(mode="after")
    def validity(self):
        if self.status == "uncalibrated":
            if self.calibration_id is not None or self.alpha is not None or self.sample_count != 0:
                raise ValueError("Inactive calibration has active metadata")
        elif (
            not self.calibration_id
            or not self.method
            or self.alpha is None
            or not 0 < self.alpha < 1
            or self.sample_count < 1
        ):
            raise ValueError("Active calibration requires provenance, alpha and independent samples")
        return self


class AtomicDecision(Model):
    atom_id: str
    base_label: FactLabel
    status: FactLabel
    probabilities: FactProb | None
    confidence_set: list[FactLabel] | None
    abstention_flag: Bool
    abstention_reasons: list[str]
    evidence_links: list[EvidenceLink]
    visual_atom_ids: list[str]
    explanation: str
    calibration: Calibration


class DecisionReport(Model):
    summary: str
    key_findings: list[str]
    unresolved_questions: list[str]
    evidence_ids: list[str]
    forensic_signal_ids: list[str]
    limitations: list[str]
    suggested_next_steps: list[str]
    misinformation_category: str | None


class HumanReview(Model):
    reviewer: str
    reviewed_at: Time
    verdict: FactLabel
    reason: str


class Decision(Model):
    run: RunInfo
    atom_set_id: str
    atomic_verdicts: list[AtomicDecision]
    base_label: FactLabel
    final_verdict: FactLabel
    probabilities: FactProb | None
    confidence_set: list[FactLabel] | None
    abstention_flag: Bool
    abstention_reasons: list[str]
    calibration: Calibration
    decision_report: DecisionReport
    human_review: HumanReview | None


class Input(Model):
    claim_text: Annotated[str, Field(min_length=1, max_length=10000)]
    language: Annotated[str, Field(pattern=r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{1,8})*$")]
    images: list[MediaRef]
    as_of: Time | None

    @model_validator(mode="after")
    def nonempty(self):
        if not self.claim_text.strip():
            raise ValueError("Caption is empty")
        if len(self.images) > 16:
            raise ValueError("At most 16 images per bundle")
        if len({m.asset_id for m in self.images}) != len(self.images):
            raise ValueError("Duplicate image assets")
        return self


def atom_set_id(bundle, atoms):
    items = [a.model_dump(mode="json") if isinstance(a, Atom) else a.copy() for a in atoms]
    for a in items:
        a["depends_on"] = sorted(a["depends_on"])
        a["spans"] = sorted(a["spans"], key=lambda x: (x["start"], x["end"]))
    return "aset_" + sha(
        canonical(
            dict(
                case_id=bundle.case_id,
                claim_revision=bundle.claim_revision,
                claim_text_sha256=sha(bundle.input.claim_text),
                image_sha256_list=sorted(m.sha256 for m in bundle.input.images),
                atomic_claims=sorted(items, key=lambda a: a["atom_id"]),
            )
        )
    )


def unique(values):
    if len(values) != len(set(values)):
        raise ValueError("Duplicate identifiers")


def references(values, valid):
    if not set(values) <= set(valid):
        raise ValueError("Dangling reference")


class AuroraBundle(Model):
    schema_version: Literal["1.0.0"]
    case_id: str
    claim_revision: PositiveInt
    mode: Mode
    created_at: Time
    input: Input
    analysis: Analysis | None
    retrieval: Retrieval | None
    decision: Decision | None
    warnings: list[Warning]
    extensions: dict[str, Any]

    @model_validator(mode="after")
    def invariants(self):
        if str(UUID(self.case_id)) != self.case_id:
            raise ValueError("Invalid case ID")

        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Non-finite extension")
            if isinstance(value, dict):
                for v in value.values():
                    finite(v)
            if isinstance(value, list):
                for v in value:
                    finite(v)

        finite(self.extensions)
        canonical(self.extensions)
        asset_ids = {m.asset_id for m in self.input.images}
        atom_ids = []
        aset = None
        for module in (self.analysis, self.retrieval, self.decision):
            if module and module.run.mode != self.mode:
                raise ValueError("Mixed demo/live mode")
        if self.analysis:
            atoms = self.analysis.atomic_claims
            if len(atoms) > 128:
                raise ValueError("At most 128 atoms per bundle")
            if not atoms:
                raise ValueError("Analysis requires atoms")
            atom_ids = [a.atom_id for a in atoms]
            unique(atom_ids)
            graph = {a.atom_id: a.depends_on for a in atoms}
            visited = set()

            def visit(aid, stack):
                if aid in visited:
                    return
                if aid in stack:
                    raise ValueError("Cyclic atom dependencies")
                for dep in graph[aid]:
                    visit(dep, stack | {aid})
                visited.add(aid)

            for atom in atoms:
                references(atom.depends_on, atom_ids)
                unique(atom.depends_on)
                if any(s.end > len(self.input.claim_text) for s in atom.spans):
                    raise ValueError("Span outside original caption")
            for aid in atom_ids:
                visit(aid, set())
            aset = atom_set_id(self, atoms)
            if aset != self.analysis.atom_set_id:
                raise ValueError("Atom set hash mismatch")
            ids = [a.atom_id for a in self.analysis.visual_assessments]
            unique(ids)
            references(ids, atom_ids)
            regions = {}
            for assessment in self.analysis.visual_assessments:
                if self.mode == "live" and assessment.inference_kind == "fixture":
                    raise ValueError("Fixture in live analysis")
                for region in assessment.supporting_regions + assessment.contradicting_regions:
                    if region.asset_id not in asset_ids:
                        raise ValueError("Region asset mismatch")
                    if region.region_id in regions and regions[region.region_id] != region:
                        raise ValueError("Conflicting region definitions")
                    if not region.region_id.startswith(
                        "rg_" + self.analysis.run.run_id.replace("-", "") + "_"
                    ):
                        raise ValueError("Region run mismatch")
                    regions[region.region_id] = region
        evidence = {}
        signals = {}
        if self.retrieval:
            if self.retrieval.atom_set_id != aset:
                raise ValueError("Stale retrieval atom set")
            for e in self.retrieval.evidence_list:
                references(e.atom_ids, atom_ids)
                if e.evidence_id in evidence or not e.evidence_id.startswith(
                    "ev_" + self.retrieval.run.run_id.replace("-", "") + "_"
                ):
                    raise ValueError("Evidence identity mismatch")
                evidence[e.evidence_id] = e
                for match in e.provenance.image_matches:
                    if match.asset_id not in asset_ids:
                        raise ValueError("Image match asset mismatch")
            for q in self.retrieval.query_log:
                references(q.atom_ids, atom_ids)
            for s in self.retrieval.forensic_signals:
                if s.signal_id in signals or not s.signal_id.startswith(
                    "fs_" + self.retrieval.run.run_id.replace("-", "") + "_"
                ):
                    raise ValueError("Forensic identity mismatch")
                signals[s.signal_id] = s
                if s.target.kind == "image" and s.target.asset_id not in asset_ids:
                    raise ValueError("Missing forensic image")
                if s.target.kind == "claim_text" and s.target.text_sha256 != sha(self.input.claim_text):
                    raise ValueError("Forensic text mismatch")
                if s.target.kind == "ocr_text":
                    text = (
                        self.extensions.get("aurora_contract", {}).get("forensic_texts", {}).get(s.signal_id)
                    )
                    if text is None or sha(text) != s.target.text_sha256:
                        raise ValueError("Missing exact OCR forensic target")
        if self.decision:
            d = self.decision
            if not self.analysis or d.atom_set_id != aset:
                raise ValueError("INPUT_ANALYSIS_REQUIRED or stale decision")
            aids = [a.atom_id for a in d.atomic_verdicts]
            unique(aids)
            references(aids, atom_ids)
            if d.run.status == "completed" and set(aids) != set(atom_ids):
                raise ValueError("Completed decision must cover every atom")
            for a in d.atomic_verdicts:
                references(a.visual_atom_ids, atom_ids)
                for link in a.evidence_links:
                    if link.atom_id != a.atom_id or link.evidence_id not in evidence:
                        raise ValueError("Invalid evidence link")
                    if link.quote is not None and link.quote not in evidence[link.evidence_id].content.text:
                        raise ValueError("Quote is not an exact substring")
            references(d.decision_report.evidence_ids, evidence)
            references(d.decision_report.forensic_signal_ids, signals)
            for item in [d, *d.atomic_verdicts]:
                label = item.final_verdict if isinstance(item, Decision) else item.status
                if item.abstention_flag and label != "InsufficientEvidence":
                    raise ValueError("Abstention requires operational IE label")
                if item.confidence_set is not None:
                    unique(item.confidence_set)
                if item.calibration.status == "uncalibrated" and (
                    item.confidence_set is not None or not item.abstention_flag
                ):
                    raise ValueError("Uncalibrated decision must abstain with null set")
                if self.mode == "live" and item.calibration.status == "demo_only":
                    raise ValueError("Demo calibration in live bundle")
        return self
