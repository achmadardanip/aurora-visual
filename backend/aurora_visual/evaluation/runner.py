import copy
import resource
import time

import numpy as np
import torch

from aurora_visual.evaluation.metrics import confusion_metrics, grounding, grouped_bootstrap


def faithfulness(model, inputs, output, method):
    """Feature deletion/insertion: an explicit intervention, not alignment-as-faithfulness."""
    base = output["probabilities"]
    labels = base.argmax(-1)
    ordering = output["transport"].coupling.sum(0).argsort(descending=True)
    n = inputs["regions"].shape[0]
    deletion, insertion = [], []
    with torch.inference_mode():
        for k in range(n + 1):
            mask = torch.ones(n, 1)
            mask[ordering[:k]] = 0
            removed = {
                **inputs,
                "regions": inputs["regions"] * mask,
                "global_feature": (inputs["regions"] * mask).mean(0),
            }
            kept = {
                **inputs,
                "regions": inputs["regions"] * (1 - mask),
                "global_feature": (inputs["regions"] * (1 - mask)).mean(0),
            }
            d = model(**removed, method=method)["probabilities"]
            ins = model(**kept, method=method)["probabilities"]
            deletion.append(float(d[torch.arange(len(labels)), labels].mean()))
            insertion.append(float(ins[torch.arange(len(labels)), labels].mean()))
    top = max(1, round(n * 0.2))
    base_score = float(base[torch.arange(len(labels)), labels].mean())
    return {
        "deletion_auc": float(np.trapezoid(deletion, dx=1 / n)),
        "insertion_auc": float(np.trapezoid(insertion, dx=1 / n)),
        "sufficiency_gap": base_score - insertion[top],
        "comprehensiveness": base_score - deletion[top],
        "scope": "feature-space intervention; global context recomputed; not pixel-level causal proof",
        "baseline": "zero embedding",
        "deletion_curve": deletion,
        "insertion_curve": insertion,
    }


def evaluate(model, samples, split="test", method="uot", confirmatory=False, interventions=True):
    selected = [s for s in samples if s["split"] == split]
    if not selected:
        raise ValueError("No samples in evaluation split")
    if confirmatory and any(s["data_kind"] == "fixture" for s in selected):
        raise ValueError("Fixtures forbidden in confirmatory evaluation")
    if confirmatory and any(
        not s.get("record", {}).get("annotations")
        or any(a["provenance"] != "human" for a in s["record"]["annotations"])
        for s in selected
    ):
        raise ValueError("Confirmatory evaluation requires human atom-level gold")
    model.eval()
    rows, faith, grounding_values, latencies = [], [], [], []
    group_correct, pairs, by_dataset = {}, {}, {}
    with torch.inference_mode():
        for sample in selected:
            started = time.perf_counter()
            output = model(**sample["inputs"], method=method)
            latencies.append((time.perf_counter() - started) * 1000)
            prediction = output["probabilities"].argmax(-1).tolist()
            truth = sample["labels"].tolist()
            if len(truth) != len(prediction) or any(y < 0 for y in truth):
                raise ValueError("Evaluation requires valid gold atom labels")
            for i, (y, p) in enumerate(zip(truth, prediction, strict=True)):
                rows.append(
                    {
                        "sample_id": sample["id"],
                        "group": sample["group"],
                        "role": sample["roles"][i],
                        "truth": y,
                        "prediction": p,
                        "pair_kind": sample.get("pair_kind"),
                    }
                )
                if sample.get("gold_regions") and sample.get("regions"):
                    idx = int(output["transport"].coupling[i].argmax())
                    value = grounding(sample["regions"][idx], sample["gold_regions"][i])
                    if value:
                        grounding_values.append(value)
            group_correct.setdefault(sample["group"], []).append(prediction == truth)
            dataset = sample.get("record", {}).get("dataset", "synthetic-smoke")
            by_dataset.setdefault(dataset, []).extend(rows[-len(truth) :])
            if "positive_inputs" in sample:
                other = model(**sample["positive_inputs"], method=method)["probabilities"].argmax(-1)
                mapping = sample["positive_correspondence"]
                pairs[sample["id"]] = float((other[mapping] == torch.tensor(prediction)).float().mean())
            if interventions:
                faith.append(faithfulness(model, sample["inputs"], output, method))
    result = confusion_metrics([r["truth"] for r in rows], [r["prediction"] for r in rows])
    roles = {r["role"] for r in rows}
    role_metrics = {
        role: confusion_metrics(
            [r["truth"] for r in rows if r["role"] == role],
            [r["prediction"] for r in rows if r["role"] == role],
        )
        for role in roles
    }
    swaps = [r for r in rows if r["pair_kind"] == "role_swap"]
    result.update(
        {
            "split": split,
            "method": method,
            "data_kind": "fixture" if any(s["data_kind"] == "fixture" for s in selected) else "research",
            "research_evaluated": confirmatory,
            "per_role": role_metrics,
            "event_bootstrap_ci": grouped_bootstrap(rows),
            "role_swap_accuracy": sum(r["truth"] == r["prediction"] for r in swaps) / len(swaps)
            if swaps
            else None,
            "hard_positive_consistency": float(np.mean(list(pairs.values()))) if pairs else None,
            "group_accuracy": float(np.mean([all(v) for v in group_correct.values()])),
            "grounding": {
                key: float(np.mean([g[key] for g in grounding_values])) for key in ("iou", "pointing")
            }
            if grounding_values
            else None,
            "faithfulness": faith,
            "per_dataset": {
                k: confusion_metrics([r["truth"] for r in v], [r["prediction"] for r in v])
                for k, v in by_dataset.items()
            },
            "inference_ms_mean": float(np.mean(latencies)),
            "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_note": "process peak: bytes on macOS; KiB on Linux",
            "rows": rows,
        }
    )
    return result


def modality_probe(samples, modality):
    result = copy.deepcopy(samples)
    for sample in result:
        for name in ("inputs", "positive_inputs"):
            if name not in sample:
                continue
            inputs = sample[name]
            if modality == "text-only":
                inputs["regions"] = torch.zeros_like(inputs["regions"])
                inputs["global_feature"] = torch.zeros_like(inputs["global_feature"])
                inputs["geometry"] = torch.zeros_like(inputs["geometry"])
                for key in ("relation_targets", "relation_mask"):
                    if key in inputs:
                        inputs[key] = torch.zeros_like(inputs[key])
                if "explicit_counter" in inputs:
                    inputs["explicit_counter"] = torch.zeros_like(inputs["explicit_counter"])
            elif modality == "image-only":
                inputs["text"] = torch.zeros_like(inputs["text"])
                if "global_text" in inputs:
                    inputs["global_text"] = torch.zeros_like(inputs["global_text"])
                inputs["atom_meta"] = torch.zeros_like(inputs["atom_meta"])
                if "relation_targets" in inputs:
                    inputs["relation_targets"] = torch.zeros_like(inputs["relation_targets"])
                if "explicit_counter" in inputs:
                    inputs["explicit_counter"] = torch.zeros_like(inputs["explicit_counter"])
            else:
                raise ValueError("Unknown modality probe")
    return result
