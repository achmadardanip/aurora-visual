from pathlib import Path

from app.models.contract import Atom, strict_json

from aurora_visual.atomization.parser import RuleAtomizer, StructuredLLMAtomizer
from aurora_visual.evaluation.metrics import extraction_metrics


def evaluate_parser(path, method="rules"):
    records = [strict_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
    reviewed = [
        r
        for r in records
        if r.get("review_status") == "reviewed"
        and r.get("gold_atoms")
        and r.get("reviewer_1")
        and r.get("reviewer_2")
    ]
    if not reviewed:
        return {
            "status": "awaiting_human_review",
            "reviewed_count": 0,
            "candidate_count": len(records),
            "metrics": None,
        }
    parser = StructuredLLMAtomizer() if method == "llm" else RuleAtomizer()
    results = []
    for record in reviewed:
        gold = [Atom.model_validate(a).model_dump(mode="json") for a in record["gold_atoms"]]
        result = parser.parse(record["caption"], record["language"])
        metrics = extraction_metrics([a.model_dump(mode="json") for a in result.atoms], gold)
        results.append(
            {
                "benchmark_id": record["benchmark_id"],
                "metrics": metrics,
                "warnings": [w.model_dump() for w in result.warnings],
            }
        )
    return {
        "status": "evaluated",
        "parser": method,
        "reviewed_count": len(reviewed),
        "candidate_count": len(records),
        "results": results,
    }
