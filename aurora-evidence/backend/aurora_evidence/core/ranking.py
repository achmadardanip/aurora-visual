"""Two-stage transparent ranking: Reciprocal Rank Fusion then feature rerank.

All weights are documented hypotheses (configurable, tuned only on validation
data). score_breakdown is preserved per evidence so any rank can be audited.
"""

import math

RRF_K = 60

# Source-kind priors are auditable metadata indicators, NOT truth.
SOURCE_KIND_PRIOR = {
    "fact_check": 0.9,
    "official": 0.8,
    "news": 0.6,
    "local_corpus": 0.5,
    "web": 0.35,
    "image_provenance": 0.5,
    "user_supplied": 0.4,
}

WEIGHTS = {
    "rrf": 0.45,
    "source_kind": 0.15,
    "temporal_eligible": 0.15,
    "atom_coverage": 0.15,
    "content_completeness": 0.10,
}

CONTENT_STATUS_COMPLETENESS = {"full": 1.0, "excerpt_only": 0.7, "snippet_only": 0.4, "unavailable": 0.0}


def reciprocal_rank_fusion(ranked_lists: list[list[str]]) -> dict[str, float]:
    """RRF over provider-ranked id lists -> {id: rrf_score}."""
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def rerank(
    items: dict[str, dict],
    rrf_scores: dict[str, float],
    as_of=None,
    group_cap: int = 2,
    limit: int = 20,
) -> list[dict]:
    """Feature-based rerank + independence-group cap for diversity.

    Mutates each item dict with relevance_score, rerank_score, score_breakdown
    and returns the ordered top list. Missing values stay None, never faked.
    """
    if rrf_scores:
        top = max(rrf_scores.values())
    else:
        top = 0.0
    for item_id, item in items.items():
        rrf = rrf_scores.get(item_id, 0.0) / top if top else 0.0
        kind = item["source"]["kind"]
        kind_prior = SOURCE_KIND_PRIOR.get(kind, 0.3)
        eligible = item["provenance"].get("temporal_eligible")
        temporal = {True: 1.0, False: 0.0, None: 0.5}[eligible]
        atom_ids = item.get("atom_ids") or []
        coverage = min(1.0, len(atom_ids) / 3) if atom_ids else 0.0
        completeness = CONTENT_STATUS_COMPLETENESS.get(item["content"]["status"], 0.0)
        breakdown = {
            "rrf": round(rrf, 6),
            "source_kind": kind_prior,
            "temporal_eligible": temporal,
            "atom_coverage": coverage,
            "content_completeness": completeness,
        }
        score = sum(WEIGHTS[name] * value for name, value in breakdown.items())
        item["score_breakdown"] = breakdown
        item["rerank_score"] = round(score, 6)
        # relevance_score in [0,1]: logistic squashing of the rerank score.
        item["relevance_score"] = round(1 / (1 + math.exp(-8 * (score - 0.5))), 6)
    ordered = sorted(items.values(), key=lambda i: (-i["rerank_score"], i["evidence_id"]))
    selected, per_group = [], {}
    for item in ordered:
        group = item["provenance"]["independence_group_id"]
        per_group[group] = per_group.get(group, 0) + 1
        if per_group[group] <= group_cap:
            selected.append(item)
        if len(selected) >= limit:
            break
    return selected
