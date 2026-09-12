import numpy as np
from scipy.optimize import linear_sum_assignment

from aurora_visual.alignment.model import LABELS


def confusion_metrics(truth, prediction):
    matrix = np.zeros((3, 3), dtype=int)
    for y, p in zip(truth, prediction, strict=True):
        matrix[y, p] += 1
    precision, recall, f1 = [], [], []
    for k in range(3):
        tp = int(matrix[k, k])
        pred = int(matrix[:, k].sum())
        total = int(matrix[k].sum())
        precision.append(tp / pred if pred else 0.0)
        recall.append(tp / total if total else 0.0)
        f1.append(2 * tp / (pred + total) if pred + total else 0.0)
    negative_count = sum(y != 1 for y in truth)
    return {
        "n": len(truth),
        "labels": LABELS,
        "confusion": matrix.tolist(),
        "macro_f1": float(np.mean(f1)),
        "per_label_f1": dict(zip(LABELS, f1)),
        "contradiction_f1": f1[1],
        "false_contradiction_rate": sum(y != 1 and p == 1 for y, p in zip(truth, prediction, strict=True))
        / negative_count
        if negative_count
        else None,
        "accuracy": sum(y == p for y, p in zip(truth, prediction, strict=True)) / len(truth)
        if truth
        else None,
    }


def grouped_bootstrap(rows, seed=17, repetitions=500):
    groups = sorted(set(r["group"] for r in rows))
    if len(groups) < 2:
        return {
            "low": None,
            "high": None,
            "units": len(groups),
            "reason": "At least two independent event/image groups required",
        }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        selected = rng.choice(groups, len(groups), replace=True)
        sample = [r for g in selected for r in rows if r["group"] == g]
        values.append(
            confusion_metrics([r["truth"] for r in sample], [r["prediction"] for r in sample])["macro_f1"]
        )
    return {
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "units": len(groups),
        "unit": "event",
        "repetitions": repetitions,
        "seed": seed,
    }


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def grounding(box, gold):
    if not gold:
        return None
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return {
        "iou": max(iou(box, b) for b in gold),
        "pointing": float(any(b[0] <= center[0] <= b[2] and b[1] <= center[1] <= b[3] for b in gold)),
    }


def extraction_metrics(predicted, gold, span_threshold=0.5):
    def token_span(atom):
        return {x for span in atom["spans"] for x in range(span["start"], span["end"])}

    def proposition(a):
        return (
            a.get("subject"),
            a["predicate"],
            a.get("object"),
            a["qualifiers"]["negated"],
            a["qualifiers"].get("quantity"),
        )

    costs = np.ones((len(predicted), len(gold)))
    for i, p in enumerate(predicted):
        for j, g in enumerate(gold):
            ps, gs = token_span(p), token_span(g)
            overlap = len(ps & gs) / max(1, len(ps | gs))
            match = p["role"] == g["role"] and overlap >= span_threshold and proposition(p) == proposition(g)
            costs[i, j] = 0 if match else 1
    a, b = linear_sum_assignment(costs)
    true_positive = sum(costs[i, j] == 0 for i, j in zip(a, b, strict=True))
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "matching": "one-to-one Hungarian; role + exact SPO/negation/quantity + Unicode-span IoU >= 0.5",
        "predicted": len(predicted),
        "gold": len(gold),
    }
