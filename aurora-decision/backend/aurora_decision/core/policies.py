"""Operational abstention policy (12 documented reasons).

Abstention changes the OPERATIONAL label to InsufficientEvidence while
base_label and the original confidence set stay stored for audit.
"""

REASONS = (
    "UNCALIBRATED",
    "CALIBRATION_MISMATCH",
    "EMPTY_PREDICTION_SET",
    "AMBIGUOUS_PREDICTION_SET",
    "INSUFFICIENT_EVIDENCE",
    "NO_ELIGIBLE_EVIDENCE",
    "CONFLICTING_EVIDENCE",
    "ESSENTIAL_ATOM_UNRESOLVED",
    "INPUT_PARTIAL",
    "TEMPORAL_UNCERTAINTY",
    "UNSUPPORTED_DOMAIN",
    "VERIFIER_UNAVAILABLE",
)


def apply_policy(
    base_label: str, confidence_set: list[str] | None, context: dict
) -> tuple[str, bool, list[str]]:
    """Return (operational_status, abstention_flag, reasons).

    context keys: calibrated (bool), calibration_mismatch (bool),
    evidence_backed (bool), conflicts (bool), verifier_available (bool).
    """
    reasons: list[str] = []
    if context.get("calibration_mismatch"):
        reasons.append("CALIBRATION_MISMATCH")
    if not context.get("calibrated"):
        # Without a matching calibration, live mode defaults to abstain.
        reasons.append("UNCALIBRATED")
        return "InsufficientEvidence", True, reasons
    if confidence_set is None:
        reasons.append("UNCALIBRATED")
        return "InsufficientEvidence", True, reasons
    if not context.get("verifier_available", True):
        reasons.append("VERIFIER_UNAVAILABLE")
    if not context.get("evidence_backed") and base_label in ("Supported", "Contradicted"):
        reasons.append("NO_ELIGIBLE_EVIDENCE")
    if context.get("conflicts"):
        reasons.append("CONFLICTING_EVIDENCE")
    if len(confidence_set) == 0:
        reasons.append("EMPTY_PREDICTION_SET")
        return "InsufficientEvidence", True, reasons
    if "InsufficientEvidence" in confidence_set and len(confidence_set) == 1:
        reasons.append("INSUFFICIENT_EVIDENCE")
        return "InsufficientEvidence", True, reasons
    if len(confidence_set) > 1:
        reasons.append("AMBIGUOUS_PREDICTION_SET")
        return "InsufficientEvidence", True, reasons
    # Singleton {Supported} or {Contradicted}: decidable if checks pass.
    if reasons:
        return "InsufficientEvidence", True, reasons
    singleton = confidence_set[0]
    return singleton, False, []
