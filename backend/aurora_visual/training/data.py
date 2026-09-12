"""Local manifest importer. Post-level labels are never converted to atom labels."""

import csv
from pathlib import Path
from typing import Literal
from uuid import uuid4

import torch
from app.models.contract import Atom, BBox, Model, Score, canonical, sha, strict_json
from PIL import Image, ImageOps
from pydantic import Field

from aurora_visual.alignment.model import LABELS
from aurora_visual.atomization.parser import RuleAtomizer
from aurora_visual.vision.features import extract, feature_identity, model_inputs


class Annotation(Model):
    atom: Atom
    label: Literal["Supported", "Contradicted", "Unobservable"] | None
    regions: list[BBox]
    provenance: Literal["human", "weak", "fixture", "unreviewed"]
    quality: Score | None


class Example(Model):
    sample_id: str
    image: str
    image_sha256: str
    caption: str
    language: str
    dataset: str
    split: Literal["train", "validation", "calibration", "test"]
    event_id: str
    source_group: str
    parent_id: str | None
    license: str
    source_url: str | None
    post_label: str | None
    annotations: list[Annotation]
    data_kind: Literal["research", "fixture"]
    extensions: dict = Field(default_factory=dict)


def load_manifest(path, require_images=True):
    path = Path(path)
    records = [
        Example.model_validate(strict_json(line)) for line in path.read_text().splitlines() if line.strip()
    ]
    seen, groups = set(), {}
    for r in records:
        if r.sample_id in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(r.sample_id)
        for key in (
            ("image", r.image_sha256),
            ("event", r.event_id),
            ("source", r.source_group),
            ("parent", r.parent_id or r.sample_id),
        ):
            if key in groups and groups[key] != r.split:
                raise ValueError(f"Split leakage: {key[0]} appears in multiple partitions")
            groups[key] = r.split
        image_path = (path.parent / r.image).resolve()
        if require_images and (not image_path.is_file() or sha(image_path.read_bytes()) != r.image_sha256):
            raise ValueError(f"Missing/mismatched image for sample {r.sample_id}")
        if not r.caption.strip() or not r.event_id or not r.source_group or not r.license:
            raise ValueError("Caption, grouping and license provenance required")
        aids = [a.atom.atom_id for a in r.annotations]
        if len(set(aids)) != len(aids):
            raise ValueError("Duplicate annotated atom ID")
        for ann in r.annotations:
            if r.data_kind == "research" and ann.provenance == "fixture":
                raise ValueError("Fixture annotation in research record")
            if ann.provenance == "unreviewed" and ann.label is not None:
                raise ValueError("Unreviewed annotation cannot supply a gold label")
            if any(s.end > len(r.caption) for s in ann.atom.spans):
                raise ValueError("Annotation span exceeds caption")
    return records


def samples_from_manifest(path, cache_dir, backbone="local-color-v1", include_unlabeled=False):
    records = load_manifest(path)
    result = []
    for record in records:
        annotations = record.annotations
        atoms = (
            [a.atom for a in annotations]
            if annotations
            else RuleAtomizer().parse(record.caption, record.language).atoms
        )
        if not atoms:
            continue
        if not include_unlabeled and (not annotations or any(a.label is None for a in annotations)):
            continue
        image_path = (Path(path).parent / record.image).resolve()
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        from app.models.contract import MediaRef

        ref = MediaRef(
            asset_id="asset_" + record.image_sha256,
            sha256=record.image_sha256,
            media_type="image/png",
            width=image.width,
            height=image.height,
            uri="dataset/image",
        )
        features = extract(
            image, ref, atoms, str(uuid4()), Path(cache_dir), backbone, 16, record.language, record.caption
        )
        inputs = model_inputs(atoms, features)
        targets = []
        for annotation in annotations:
            if annotation.regions:
                from aurora_visual.evaluation.metrics import iou

                targets.append(
                    max(
                        range(len(features["regions"])),
                        key=lambda i: max(iou(features["regions"][i].bbox, b) for b in annotation.regions),
                    )
                )
            else:
                targets.append(-1)
        result.append(
            {
                "id": record.sample_id,
                "group": record.event_id,
                "source_group": record.source_group,
                "parent_id": record.parent_id or record.sample_id,
                "image_sha256": record.image_sha256,
                "split": record.split,
                "data_kind": record.data_kind,
                "feature_config": feature_identity(features["config"]),
                "inputs": inputs,
                "labels": torch.tensor(
                    [LABELS.index(a.label) if a.label else -1 for a in annotations], dtype=torch.long
                ),
                "region_targets": torch.tensor(targets, dtype=torch.long),
                "roles": [a.role for a in atoms],
                "gold_regions": [a.regions for a in annotations],
                "regions": [r.bbox for r in features["regions"]],
                "atoms": [a.model_dump(mode="json") for a in atoms],
                "record": record.model_dump(mode="json"),
                "pair_id": record.extensions.get("pair_id"),
                "pair_kind": record.extensions.get("pair_kind"),
            }
        )
    by_id = {sample["id"]: sample for sample in result}
    for sample in result:
        ext = sample["record"]["extensions"]
        positive_id = ext.get("hard_positive_id")
        if positive_id:
            positive = by_id.get(positive_id)
            if (
                positive is None
                or positive["split"] != sample["split"]
                or positive["image_sha256"] != sample["image_sha256"]
            ):
                raise ValueError("Hard positive requires the same image, proposals and split")
            mapping = ext.get("atom_correspondence")
            if (
                not isinstance(mapping, list)
                or sorted(mapping) != list(range(len(sample["labels"])))
                or len(positive["labels"]) != len(mapping)
            ):
                raise ValueError("Hard-positive atom correspondence must be explicit and bijective")
            mapping = torch.tensor(mapping, dtype=torch.long)
            if not torch.equal(sample["labels"], positive["labels"][mapping]):
                raise ValueError("Hard-positive labels disagree after correspondence")
            sample["positive_inputs"] = positive["inputs"]
            sample["positive_correspondence"] = mapping
    return result


def import_public(source, source_path, output, image_root, split, license_note, groups_path, visualnews=None):
    """Normalize verified release formats; group assignments supplied by the researcher."""
    groups = strict_json(Path(groups_path).read_bytes())
    if source == "verite":
        with Path(source_path).open() as f:
            rows = list(csv.DictReader(f))
    else:
        data = strict_json(Path(source_path).read_bytes())
        rows = data["annotations"] if source == "newsclippings" else data
    if source == "newsclippings":
        if not visualnews:
            raise ValueError("NewsCLIPpings requires VisualNews origin/data.json")
        vn = {str(r["id"]): r for r in strict_json(Path(visualnews).read_bytes())}
    if source == "cosmos":
        expanded = []
        for index, row in enumerate(rows):
            captions = (
                [a["caption"] for a in row["articles"]]
                if "articles" in row
                else [row["caption1"], row["caption2"]]
            )
            for caption_index, caption in enumerate(captions):
                expanded.append(
                    {
                        "caption": caption,
                        "image": row["img_local_path"],
                        "release_row": index,
                        "caption_index": caption_index,
                        "pair_context_label": row.get("label"),
                    }
                )
        rows = expanded
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(rows):
        if source == "newsclippings":
            caption, image = vn[str(row["id"])]["caption"], vn[str(row["image_id"])]["image_path"]
            post = str(row["falsified"])
        elif source == "verite":
            caption, image, post = row["caption"], row["image_path"], row["label"]
        elif source == "mmfakebench":
            caption, image, post = row["text"], row["image_path"].lstrip("/"), row["gt_answers"]
        elif source == "cosmos":
            caption, image, post = row["caption"], row["image"], None
        else:
            raise ValueError("Unsupported source")
        group = groups[str(row.get("release_row", i))]
        root = Path(image_root).resolve()
        relative_image = image[2:] if image.startswith("./") else image
        if Path(relative_image).is_absolute() or ".." in Path(relative_image).parts:
            raise ValueError("Dataset path escapes image root")
        image_path = (root / relative_image).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError("Dataset path escapes image root")
        import os

        records.append(
            Example(
                sample_id=f"{source}-{i}",
                image=os.path.relpath(image_path, output.parent),
                image_sha256=sha(image_path.read_bytes()),
                caption=caption,
                language="en",
                dataset=source,
                split=split,
                event_id=group["event_id"],
                source_group=group["source_group"],
                parent_id=group.get("parent_id"),
                license=license_note,
                source_url=group.get("source_url"),
                post_label=str(post) if post is not None else None,
                annotations=[],
                data_kind="research",
                extensions={
                    "label_mapping": "post-level only; no atomic supervision",
                    "release_row": row.get("release_row", i),
                    "pair_context_label": row.get("pair_context_label"),
                },
            )
        )
    output.write_text("\n".join(r.model_dump_json() for r in records) + "\n")
    return {"count": len(records), "manifest_sha256": sha(output.read_bytes()), "atomic_gold": 0}


def smoke_samples(seed=17):
    """Engineering fixture features, deliberately not a public dataset or research result."""
    generator = torch.Generator().manual_seed(seed)
    result = []
    for i in range(8):
        regions = torch.rand((4, 12), generator=generator)
        text = torch.stack([regions[0], 1 - regions[1], torch.zeros(12)])
        meta = torch.zeros((3, 13))
        meta[0, 3] = meta[1, 3] = meta[2, 5] = 1
        meta[2, -2] = 1
        geometry = torch.tensor(
            [
                [0.0, 0.0, 0.5, 0.5, 1.0],
                [0.5, 0.0, 1.0, 0.5, 1.0],
                [0.0, 0.5, 0.5, 1.0, 1.0],
                [0.5, 0.5, 1.0, 1.0, 1.0],
            ]
        )
        inputs = dict(
            text=text,
            global_text=text.mean(0),
            regions=regions,
            atom_meta=meta,
            geometry=geometry,
            global_feature=regions.mean(0),
            relation_targets=torch.tensor([[0.25, 0.25], [0.75, 0.25], [0.5, 0.5]]),
            explicit_counter=torch.tensor([0.0, 1.0, 0.0]),
        )
        positive = {k: v.clone() for k, v in inputs.items()}
        positive["text"] = positive["text"] + torch.randn(text.shape, generator=generator) * 0.005
        result.append(
            {
                "id": f"fixture-{i}",
                "group": f"event-{i}",
                "source_group": f"source-{i}",
                "parent_id": f"fixture-{i}",
                "image_sha256": sha(f"synthetic-feature-image-{i}"),
                "split": "train"
                if i < 5
                else "validation"
                if i == 5
                else "calibration"
                if i == 6
                else "test",
                "data_kind": "fixture",
                "feature_config": {"backbone": "synthetic-feature-fixture-v1", "embedding_dim": 12},
                "inputs": inputs,
                "labels": torch.tensor([0, 1, 2]),
                "region_targets": torch.tensor([0, 1, -1]),
                "roles": ["attribute", "attribute", "time"],
                "positive_inputs": positive,
                "positive_correspondence": torch.arange(3),
                "pair_kind": "hard_positive",
                "pair_id": f"pair-{i}",
            }
        )
    return result


def dataset_hash(samples):
    def tensor_hashes(inputs):
        return {
            key: {
                "sha256": sha(value.detach().cpu().contiguous().numpy().tobytes()),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in inputs.items()
        }

    return sha(
        canonical(
            [
                {
                    "id": s["id"],
                    "split": s["split"],
                    "group": s["group"],
                    "source_group": s["source_group"],
                    "parent_id": s["parent_id"],
                    "image": s["image_sha256"],
                    "data_kind": s["data_kind"],
                    "feature_config": s["feature_config"],
                    "labels": s["labels"].tolist(),
                    "region_targets": s["region_targets"].tolist(),
                    "roles": s["roles"],
                    "feature_hashes": tensor_hashes(s["inputs"]),
                    "positive_feature_hashes": tensor_hashes(s.get("positive_inputs", {})),
                    "positive_correspondence": s["positive_correspondence"].tolist()
                    if "positive_correspondence" in s
                    else None,
                }
                for s in samples
            ]
        )
    )
