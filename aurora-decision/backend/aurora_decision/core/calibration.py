"""Split conformal prediction: global and Mondrian (role, label) groups.

Exact finite-sample order statistics with <= for ties; k = ceil((n+1)(1-alpha))
1-indexed; q=+infinity when n_g=0 or k_g>n_g. JSON-safe infinity sentinel
{kind:"infinity", value:null}. Calibration artifacts bind to input route,
feature schema, label map, and alpha — mismatch means uncalibrated + warning,
never silently reusing stale quantiles.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from aurora_decision.core.fusion import FACT_LABELS


def safe_quantile_representation(q: float) -> dict:
    """JSON-safe representation; +inf becomes the structured sentinel."""
    if math.isinf(q):
        return {"kind": "infinity", "value": None}
    return {"kind": "finite", "value": q}


@dataclass
class CalibrationArtifact:
    calibration_id: str
    method: str  # "split_conformal_global" | "split_conformal_mondrian"
    alpha: float
    input_route: str  # visual_only | evidence_only | fused
    label_map: list[str]
    feature_schema_sha: str
    groups: dict  # group_key -> {n, k, q(float or inf), samples: [ids]}
    created_at: str
    sample_count: int
    fallback: str | None  # e.g. "global" when small groups fall back

    def to_json(self) -> dict:
        return {
            "calibration_id": self.calibration_id,
            "method": self.method,
            "alpha": self.alpha,
            "input_route": self.input_route,
            "label_map": self.label_map,
            "feature_schema_sha": self.feature_schema_sha,
            "sample_count": self.sample_count,
            "fallback": self.fallback,
            "created_at": self.created_at,
            "groups": {
                key: {
                    "n": value["n"],
                    "k": value["k"],
                    "q": safe_quantile_representation(value["q"]),
                    "sample_ids": value["samples"],
                }
                for key, value in self.groups.items()
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> "CalibrationArtifact":
        groups = {}
        for key, value in data["groups"].items():
            q = math.inf if value["q"]["kind"] == "infinity" else value["q"]["value"]
            groups[key] = {"n": value["n"], "k": value["k"], "q": q, "samples": value.get("sample_ids", [])}
        return cls(
            calibration_id=data["calibration_id"],
            method=data["method"],
            alpha=data["alpha"],
            input_route=data["input_route"],
            label_map=data["label_map"],
            feature_schema_sha=data["feature_schema_sha"],
            groups=groups,
            created_at=data["created_at"],
            sample_count=data["sample_count"],
            fallback=data.get("fallback"),
        )


def conformal_quantile(scores: list[float], alpha: float) -> tuple[float, int, int]:
    """k-th (1-indexed) order statistic with k=ceil((n+1)(1-alpha)); inf if k>n."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n or k < 1:
        return math.inf, k, n
    ordered = sorted(scores)
    return ordered[k - 1], k, n


def calibrate(
    calibration_samples: list[dict],
    alpha: float,
    input_route: str,
    feature_schema_sha: str,
    method: str = "split_conformal_mondrian",
    created_at: str = "",
) -> CalibrationArtifact:
    """calibration_samples: [{score: 1-p(y|x), group: (role, label), id}].

    Scores use the true label at calibration time; groups are (role, label).
    A deterministic "one atom per event" rule is the caller's responsibility —
    sample_count counts independent units, not every atom.
    """
    groups: dict[str, dict] = {}
    global_scores = []
    for sample in calibration_samples:
        score = float(sample["score"])
        if not math.isfinite(score):
            raise ValueError("Skor kalibrasi tidak finite")
        global_scores.append(score)
        if method == "split_conformal_mondrian":
            key = f"{sample['group'][0]}|{sample['group'][1]}"
            groups.setdefault(key, {"scores": [], "samples": []})
            groups[key]["scores"].append(score)
            groups[key]["samples"].append(sample["id"])
    if method == "split_conformal_mondrian":
        rendered = {}
        for key, value in groups.items():
            q, k, n = conformal_quantile(value["scores"], alpha)
            rendered[key] = {"n": n, "k": k, "q": q, "samples": value["samples"]}
    else:
        q, k, n = conformal_quantile(global_scores, alpha)
        rendered = {"global": {"n": n, "k": k, "q": q, "samples": [s["id"] for s in calibration_samples]}}
    return CalibrationArtifact(
        calibration_id="cal_" + uuid4().hex,
        method=method,
        alpha=alpha,
        input_route=input_route,
        label_map=list(FACT_LABELS),
        feature_schema_sha=feature_schema_sha,
        groups=rendered,
        created_at=created_at,
        sample_count=len(calibration_samples),
        fallback=None,
    )


class CalibrationStore:
    """Artifact persistence: save/load/alpha change -> new artifact required."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, artifact: CalibrationArtifact) -> Path:
        path = self.directory / f"{artifact.calibration_id}.json"
        path.write_text(json.dumps(artifact.to_json(), indent=2, allow_nan=False))
        return path

    def list(self) -> list[dict]:
        return [json.loads(path.read_text()) for path in sorted(self.directory.glob("cal_*.json"))]

    def load(self, calibration_id: str) -> CalibrationArtifact | None:
        path = self.directory / f"{calibration_id}.json"
        if not path.is_file():
            return None
        return CalibrationArtifact.from_json(json.loads(path.read_text()))


def prediction_set(
    probabilities: dict[str, float],
    role: str,
    artifact: CalibrationArtifact,
) -> tuple[list[str], dict]:
    """C(x) = {z : 1 - p(z|x) <= q_group(role, z)} with group/fallback lookup."""
    diagnostics = {}
    members = []
    for label in artifact.label_map:
        score = 1.0 - probabilities.get(label, 0.0)
        key = f"{role}|{label}"
        entry = artifact.groups.get(key)
        used = key
        if entry is None or entry["n"] == 0:
            # Pre-declared fallback: global group (no conditional guarantee).
            entry = artifact.groups.get("global")
            used = "global"
        if entry is None:
            diagnostics[label] = {"group": used, "n": 0, "k": None, "q": "infinity", "fallback": True}
            continue
        q = entry["q"]
        included = score <= q
        diagnostics[label] = {
            "group": used,
            "n": entry["n"],
            "k": entry["k"],
            "q": safe_quantile_representation(q),
            "score": round(score, 6),
            "fallback": used != key,
        }
        if included:
            members.append(label)
    return members, diagnostics


def matches_route(
    artifact: CalibrationArtifact, input_route: str, feature_schema_sha: str, label_map: list[str]
) -> tuple[bool, str | None]:
    """Artifact binding check; mismatch -> uncalibrated, never silent reuse."""
    if artifact.input_route != input_route:
        return False, f"input_route {artifact.input_route} != {input_route}"
    if artifact.feature_schema_sha != feature_schema_sha:
        return False, "feature schema berubah"
    if artifact.label_map != label_map:
        return False, "label map berubah"
    return True, None
