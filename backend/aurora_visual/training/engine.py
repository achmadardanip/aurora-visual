import json
import random
from pathlib import Path

import numpy as np
import torch
from app.models.contract import canonical, sha
from torch.nn import functional as F

from aurora_visual.alignment.model import LABELS, VisualHead
from aurora_visual.training.data import dataset_hash

DEFAULT = {
    "epochs": 5,
    "learning_rate": 0.003,
    "seed": 17,
    "hidden": 32,
    "method": "uot",
    "use_unmatched": True,
    "cosine_only": False,
    "hard_positives": True,
    "loss_weights": {"ce": 1.0, "contrastive": 0.1, "margin": 0.1, "consistency": 0.2, "entropy": 0.001},
}


def loss_terms(output, sample, class_weights, positive_output=None):
    labels = sample["labels"]
    ce = F.cross_entropy(output["logits"], labels, weight=class_weights)
    targets = sample["region_targets"]
    valid = (targets >= 0) & (labels == 0)
    contrastive = F.cross_entropy(-output["cost"][valid] / 0.1, targets[valid]) if valid.any() else ce * 0
    support, contra = labels == 0, labels == 1
    margin = (
        F.relu(
            0.2 + output["cost"][support].min(1).values.mean() - output["cost"][contra].min(1).values.mean()
        )
        if support.any() and contra.any()
        else ce * 0
    )
    consistency = ce * 0
    if positive_output is not None:
        mapping = sample["positive_correspondence"]
        p, q = output["probabilities"], positive_output["probabilities"][mapping]
        prob_loss = 0.5 * (
            (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(-1).mean()
            + (q * (q.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(-1).mean()
        )
        pi, qi = output["transport"].coupling, positive_output["transport"].coupling[mapping]
        # Region columns identical: same image, same cached proposals; atom rows explicitly mapped.
        consistency = prob_loss + F.mse_loss(pi, qi)
    entropy = -output["entropy"].mean()
    return {
        "ce": ce,
        "contrastive": contrastive,
        "margin": margin,
        "consistency": consistency,
        "entropy": entropy,
    }


def save_checkpoint(path, model, optimizer, metadata, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "metadata": metadata,
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
    }
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {**metadata, "checkpoint_sha256": sha(path.read_bytes()), "history": history},
            indent=2,
            allow_nan=False,
        )
    )


def load_checkpoint(path, for_live=False):
    path = Path(path)
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise ValueError("CHECKPOINT_MISSING")
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if sha(path.read_bytes()) != sidecar.get("checkpoint_sha256"):
        raise ValueError("CHECKPOINT_HASH_MISMATCH")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    required = {
        "config",
        "seed",
        "backbone_version",
        "label_map",
        "data_hash",
        "split_hash",
        "split_ids",
        "split_groups",
        "feature_config",
        "data_kind",
        "embedding_dim",
        "trained_steps",
        "epoch",
    }
    if (
        payload.get("format_version") != 1
        or not required <= set(meta)
        or meta["label_map"] != LABELS
        or meta["trained_steps"] < 1
    ):
        raise ValueError("CHECKPOINT_METADATA_INVALID")
    if (
        meta["split_hash"] != sha(canonical(meta["split_ids"]))
        or not meta["split_ids"].get("train")
        or not meta["split_ids"].get("validation")
    ):
        raise ValueError("CHECKPOINT_SPLIT_METADATA_INVALID")
    ids = [item for partition in meta["split_ids"].values() for item in partition]
    if len(ids) != len(set(ids)) or len(meta["data_hash"]) != 64:
        raise ValueError("CHECKPOINT_SPLIT_METADATA_INVALID")
    if any(not torch.isfinite(value).all() for value in payload["model_state"].values()):
        raise ValueError("CHECKPOINT_NONFINITE")
    if for_live and meta["data_kind"] != "research":
        raise ValueError("CHECKPOINT_FIXTURE_ONLY")
    if not isinstance(meta["feature_config"], dict) or not meta["feature_config"].get("backbone"):
        raise ValueError("CHECKPOINT_FEATURE_METADATA_INVALID")
    groups = {}
    if set(meta["split_groups"]) != set(meta["split_ids"]):
        raise ValueError("CHECKPOINT_SPLIT_METADATA_INVALID")
    for partition, rows in meta["split_groups"].items():
        if len(rows) != len(meta["split_ids"][partition]):
            raise ValueError("CHECKPOINT_SPLIT_METADATA_INVALID")
        for row in rows:
            for kind in ("group", "source_group", "image_sha256", "parent_id"):
                if not row.get(kind):
                    raise ValueError("CHECKPOINT_SPLIT_METADATA_INVALID")
                key = (kind, row[kind])
                if key in groups and groups[key] != partition:
                    raise ValueError("CHECKPOINT_SPLIT_LEAKAGE")
                groups[key] = partition
    for key in required:
        if sidecar.get(key) != meta[key]:
            raise ValueError("CHECKPOINT_SIDECAR_MISMATCH")
    config = meta["config"]
    model = VisualHead(
        meta["embedding_dim"], config["hidden"], config["use_unmatched"], config["cosine_only"]
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, meta, payload


def check_feature_compatibility(metadata, samples):
    if any(sample["feature_config"] != metadata["feature_config"] for sample in samples):
        raise ValueError("CHECKPOINT_FEATURE_MISMATCH: encoder, weights, preprocessing or grid changed")


def train(samples, output, config=None, backbone="local-color-v1", resume=None):
    config = {**DEFAULT, **(config or {})}
    config["loss_weights"] = {**DEFAULT["loss_weights"], **config.get("loss_weights", {})}
    if type(config["epochs"]) is not int or config["epochs"] < 1:
        raise ValueError("Epochs must be a positive integer")
    feature_configs = [sample["feature_config"] for sample in samples]
    if not feature_configs or any(value != feature_configs[0] for value in feature_configs):
        raise ValueError("Training requires one consistent encoder/preprocessing configuration")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(1)
    seen_groups = {}
    for sample in samples:
        for kind in ("group", "source_group", "image_sha256", "parent_id"):
            key = (kind, sample[kind])
            if key in seen_groups and seen_groups[key] != sample["split"]:
                raise ValueError("Split leakage in training samples")
            seen_groups[key] = sample["split"]
    train_rows = [s for s in samples if s["split"] == "train"]
    validation = [s for s in samples if s["split"] == "validation"]
    if not train_rows or not validation:
        raise ValueError("Training requires separate train and validation partitions")
    if any((s["labels"] < 0).any() for s in train_rows + validation):
        raise ValueError("Missing atomic labels; post-level labels are not gold")
    dim = train_rows[0]["inputs"]["text"].shape[-1]
    model = VisualHead(dim, config["hidden"], config["use_unmatched"], config["cosine_only"])
    history, steps, start = [], 0, 0
    d_hash = dataset_hash(samples)
    splits = {
        split: [s["id"] for s in samples if s["split"] == split]
        for split in ("train", "validation", "calibration", "test")
    }
    if resume:
        model, meta, state = load_checkpoint(resume)
        if (
            meta["backbone_version"] != backbone
            or meta["data_hash"] != d_hash
            or any(
                meta["config"][k] != config[k]
                for k in (
                    "hidden",
                    "use_unmatched",
                    "cosine_only",
                    "method",
                    "hard_positives",
                    "loss_weights",
                    "seed",
                    "learning_rate",
                )
            )
        ):
            raise ValueError("Resume data/config mismatch")
        if config["epochs"] <= meta["epoch"]:
            raise ValueError("Resume epochs must exceed the saved epoch")
        steps, start, history = meta["trained_steps"], meta["epoch"], state["history"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    if resume:
        optimizer.load_state_dict(state["optimizer_state"])
        torch.set_rng_state(state["torch_rng_state"])
    counts = torch.bincount(torch.cat([s["labels"] for s in train_rows]), minlength=3).float()
    weights = (counts.sum() / counts.clamp_min(1)) / 3
    grad_norms = {}
    for epoch in range(start, config["epochs"]):
        model.train()
        totals = {k: 0.0 for k in config["loss_weights"]}
        loss_sum, nonconverged = 0.0, 0
        for sample in train_rows:
            optimizer.zero_grad()
            output_values = model(**sample["inputs"], method=config["method"])
            positive = (
                model(**sample["positive_inputs"], method=config["method"])
                if config["hard_positives"] and "positive_inputs" in sample
                else None
            )
            terms = loss_terms(output_values, sample, weights, positive)
            loss = sum(terms[k] * config["loss_weights"][k] for k in terms)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        raise ValueError("Non-finite gradients")
                    grad_norms[name] = max(grad_norms.get(name, 0), float(param.grad.norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach())
            nonconverged += int(not output_values["transport"].diagnostics["converged"])
            for k, v in terms.items():
                totals[k] += float(v.detach()) / len(train_rows)
        model.eval()
        with torch.inference_mode():
            val_loss = sum(
                float(F.cross_entropy(model(**s["inputs"], method=config["method"])["logits"], s["labels"]))
                for s in validation
            ) / len(validation)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / len(train_rows),
                "validation_ce": val_loss,
                "loss_terms": totals,
                "nonconverged_batches": nonconverged,
            }
        )
    if steps == 0:
        raise ValueError("No training steps")
    metadata = {
        "config": config,
        "seed": config["seed"],
        "backbone_version": backbone,
        "feature_config": feature_configs[0],
        "label_map": LABELS,
        "data_hash": d_hash,
        "split_hash": sha(canonical(splits)),
        "split_ids": splits,
        "split_groups": {
            partition: [
                {key: sample[key] for key in ("group", "source_group", "image_sha256", "parent_id")}
                for sample in samples
                if sample["split"] == partition
            ]
            for partition in splits
        },
        "data_kind": "fixture" if any(s["data_kind"] == "fixture" for s in samples) else "research",
        "embedding_dim": dim,
        "trained_steps": steps,
        "epoch": config["epochs"],
        "gradient_norms": grad_norms,
        "calibration_status": "uncalibrated",
        "research_evaluated": False,
        "optimizer": "AdamW",
        "selection": "loss weights provided externally; tune on validation only; no test tuning",
    }
    save_checkpoint(output, model, optimizer, metadata, history)
    return metadata, history
