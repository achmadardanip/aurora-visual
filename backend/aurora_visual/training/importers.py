"""Importers for capability datasets.

Two manifest kinds:

- Caption-bearing datasets (snli-ve, countbench, refcoco, flickr30k-entities,
  visual-genome) become standard ``Example`` manifests. SNLI-VE additionally
  emits *weak* atom-level S/C/U labels from its pair-level visual-entailment
  label (entailment->Supported, contradiction->Contradicted, neutral->
  Unobservable). The mapping is recorded in extensions; weak labels are never
  gold and never enter confirmatory evaluation.
- Captionless/templated capability datasets (sa1b, fsc-147, im2gps3k, osv5m,
  vggface2, yfcc100m) become ``CapabilityExample`` manifests: images plus
  capability payload (masks/counts/gps/identity/capture-time) for grounding and
  capability evaluation. They carry no captions and no S/C/U supervision.

All importers require locally obtained files (no downloads), an explicit
license note, and keep path safety: no absolute paths, no ``..``.
"""

import csv
import json
import os
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree

from app.models.contract import Model, sha, strict_json
from pydantic import Field

from aurora_visual.atomization.parser import RuleAtomizer
from aurora_visual.training.data import Example

CAPTION_SOURCES = ("snli-ve", "countbench", "refcoco", "flickr30k-entities", "visual-genome")
CAPABILITY_SOURCES = ("sa1b", "fsc-147", "im2gps3k", "osv5m", "vggface2", "yfcc100m")

SNLI_VE_LABELS = {
    "entailment": "Supported",
    "contradiction": "Contradicted",
    "neutral": "Unobservable",
}


class CapabilityExample(Model):
    sample_id: str
    image: str
    image_sha256: str
    dataset: str
    split: Literal["train", "validation", "calibration", "test"]
    event_id: str
    source_group: str
    license: str
    source_url: str | None
    data_kind: Literal["research", "fixture"] = "research"
    extensions: dict = Field(default_factory=dict)


def _resolve_image(root: Path, relative: str, output_dir: Path) -> tuple[str, str]:
    relative = relative[2:] if relative.startswith("./") else relative
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Dataset path escapes image root: {relative}")
    image_path = (root / candidate).resolve()
    if not image_path.is_relative_to(root) or not image_path.is_file():
        raise ValueError(f"Image missing inside root: {relative}")
    return os.path.relpath(image_path, output_dir), sha(image_path.read_bytes())


def _groups_or_default(groups_path, rows):
    """Group map keyed by release row index; defaults to image-stem grouping."""
    if groups_path:
        data = strict_json(Path(groups_path).read_bytes())
        if not isinstance(data, dict):
            raise ValueError("groups file must map row index to group fields")
        return lambda index, row: data[str(index)]
    return lambda index, row: {
        "event_id": Path(row["image"]).stem,
        "source_group": row.get("dataset", "dataset"),
        "parent_id": None,
        "source_url": None,
    }


def _write_manifest(records, output) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(r.model_dump_json() for r in records) + "\n")
    return {
        "count": len(records),
        "manifest_sha256": sha(output.read_bytes()),
        "atomic_gold": 0,
        "note": "Capability/caption manifest; see extensions for labels and capability payloads",
    }


def import_caption_dataset(
    source,
    source_path,
    output,
    image_root,
    split,
    license_note,
    groups_path=None,
    relations_path=None,
    attributes_path=None,
    boxes_dir=None,
) -> dict:
    """Normalize caption-bearing capability datasets into Example manifests."""
    if source not in CAPTION_SOURCES:
        raise ValueError(f"Unsupported caption source: {source}")
    source_file = Path(source_path)
    rows = _load_rows(source, source_file, relations_path, attributes_path, boxes_dir)
    output = Path(output)
    group_of = _groups_or_default(groups_path, rows)
    root = Path(image_root).resolve()
    parser = RuleAtomizer()
    records, weak_labels = [], 0
    for index, row in enumerate(rows):
        image_rel, image_sha = _resolve_image(root, row["image"], output.parent)
        caption = str(row["caption"]).strip()
        if not caption:
            raise ValueError(f"Empty caption at row {index}")
        annotations = []
        if source == "snli-ve":
            label = SNLI_VE_LABELS.get(str(row.get("label", "")).strip().lower())
            if label is None:
                raise ValueError(f"Unknown SNLI-VE label at row {index}: {row.get('label')!r}")
            for atom in parser.parse(caption, "en").atoms:
                annotations.append(
                    {
                        "atom": atom,
                        "label": label,
                        "regions": [],
                        "provenance": "weak",
                        "quality": None,
                    }
                )
                weak_labels += 1
        group = group_of(index, row)
        extensions = {
            "release_row": index,
            "label_mapping": (
                "snli-ve pair label mapped entailment->Supported, contradiction->Contradicted, "
                "neutral->Unobservable; weak provenance, all atoms inherit the pair label; "
                "not gold, excluded from confirmatory evaluation"
                if source == "snli-ve"
                else "post-level only; no atomic supervision"
            ),
        }
        for key in ("count", "grounding", "regions", "relations", "attributes"):
            if row.get(key) is not None:
                extensions[key] = row[key]
        if row.get("post_label") is not None:
            extensions["post_label"] = row["post_label"]
        records.append(
            Example(
                # Content-derived ID: separately imported split files cannot
                # collide on row indices, and a true double import is flagged.
                sample_id=f"{source}-{sha(image_rel + caption)[:16]}",
                image=image_rel,
                image_sha256=image_sha,
                caption=caption,
                language=str(row.get("language", "en")),
                dataset=source,
                split=split,
                event_id=str(group["event_id"]),
                source_group=str(group["source_group"]),
                parent_id=group.get("parent_id"),
                license=license_note,
                source_url=group.get("source_url"),
                post_label=str(row["post_label"]) if row.get("post_label") is not None else None,
                annotations=annotations,
                data_kind="research",
                extensions=extensions,
            )
        )
    result = _write_manifest(records, output)
    result["weak_atom_labels"] = weak_labels
    return result


def _load_rows(source, source_file, relations_path=None, attributes_path=None, boxes_dir=None):
    if source == "snli-ve":
        rows = []
        for line in source_file.read_text().splitlines():
            if line.strip():
                entry = strict_json(line.encode())
                rows.append(
                    {
                        "caption": entry.get("caption") or entry.get("hypothesis") or "",
                        "image": entry.get("image") or entry.get("image_path") or "",
                        "label": entry.get("label"),
                        "post_label": entry.get("label"),
                        "language": "en",
                    }
                )
        return rows
    if source == "countbench":
        data = strict_json(source_file.read_bytes())
        entries = data.get("entries", data) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError("CountBench input must be a list or {'entries': [...]}")
        rows = []
        for entry in entries:
            image = entry.get("image_path") or entry.get("image")
            if not image:
                raise ValueError("CountBench entry needs a local image_path after mirroring")
            rows.append(
                {
                    "caption": entry.get("caption", ""),
                    "image": image,
                    "count": {"number": entry.get("number"), "source": "countbench-curated"},
                    "language": "en",
                }
            )
        return rows
    if source == "refcoco":
        data = strict_json(source_file.read_bytes())
        if not isinstance(data, list):
            raise ValueError("RefCOCO input must be a JSON list of prepared reference rows")
        rows = []
        for entry in data:
            bbox = entry.get("bbox")
            width, height = entry.get("width"), entry.get("height")
            if not bbox or not width or not height:
                raise ValueError("RefCOCO rows need bbox [x,y,w,h] and image width/height")
            x, y, w, h = (float(v) for v in bbox)
            rows.append(
                {
                    "caption": entry.get("sentence") or entry.get("sent") or "",
                    "image": entry.get("image") or entry.get("image_path") or "",
                    "grounding": {
                        "phrase": entry.get("sentence") or entry.get("sent") or "",
                        "bbox": [
                            x / width,
                            y / height,
                            (x + w) / width,
                            (y + h) / height,
                        ],
                        "provenance": "refcoco-gold-region",
                    },
                    "language": "en",
                }
            )
        return rows
    if source == "flickr30k-entities":
        data = strict_json(source_file.read_bytes())
        boxes = _flickr30k_boxes(boxes_dir) if boxes_dir else {}
        rows = []
        for image_id, sentences in data.items():
            for sentence in sentences:
                phrases = [
                    {"phrase": p["phrase"], "bbox": boxes.get(p["phrase_id"])}
                    for p in sentence.get("phrases", [])
                ]
                rows.append(
                    {
                        "caption": sentence.get("sentence", ""),
                        "image": entry_image_path(image_id),
                        "grounding": {"phrases": phrases[:32]},
                        "language": "en",
                    }
                )
        return rows
    if source == "visual-genome":
        data = strict_json(source_file.read_bytes())
        relations = _vg_by_image(relations_path, "relationships") if relations_path else {}
        attributes = _vg_by_image(attributes_path, "attributes") if attributes_path else {}
        rows, seen = [], {}
        for entry in data:
            image = entry.get("image") or {}
            image_id = str(image.get("image_id", entry.get("image_id", "")))
            if not image_id:
                raise ValueError("Visual Genome region rows need image_id")
            relative = str(image.get("url", "")).rsplit("/", 1)[-1] if image.get("url") else None
            if image_id in seen:
                seen[image_id]["regions"].append(_vg_region(entry, image))
                continue
            row = {
                "caption": entry.get("phrase", ""),
                "image": relative or f"{image_id}.jpg",
                "dataset": "visual-genome",
                "regions": [_vg_region(entry, image)],
                "relations": relations.get(image_id),
                "attributes": attributes.get(image_id),
                "language": "en",
            }
            seen[image_id] = row
            rows.append(row)
        return rows
    raise ValueError(f"Unsupported source: {source}")


def entry_image_path(image_id: str) -> str:
    return f"{image_id}.jpg"


def _vg_region(entry, image):
    region = entry.get("region") or {}
    width, height = image.get("width") or 1, image.get("height") or 1
    x, y = region.get("x", 0), region.get("y", 0)
    w, h = region.get("width", 0), region.get("height", 0)
    return {
        "phrase": entry.get("phrase", ""),
        "bbox": [x / width, y / height, (x + w) / width, (y + h) / height],
        "region_id": entry.get("region_id"),
    }


def _vg_by_image(path, key):
    data = strict_json(Path(path).read_bytes())
    by_image = {}
    for entry in data if isinstance(data, list) else data.get(key, []):
        image_id = str((entry.get("image") or {}).get("image_id", entry.get("image_id", "")))
        by_image.setdefault(image_id, []).append(entry)
    for image_id, entries in by_image.items():
        by_image[image_id] = entries[:64]
    return by_image


def _flickr30k_boxes(boxes_dir):
    boxes = {}
    for xml_path in sorted(Path(boxes_dir).glob("*.xml"))[:100_000]:
        try:
            root = ElementTree.fromstring(xml_path.read_text())
        except ElementTree.ParseError:
            continue
        for obj in root.iter("object"):
            name = obj.findtext("name")
            bnd = obj.find("bndbox")
            if name is None or bnd is None:
                continue
            try:
                boxes[name] = [
                    int(bnd.findtext("xmin")),
                    int(bnd.findtext("ymin")),
                    int(bnd.findtext("xmax")),
                    int(bnd.findtext("ymax")),
                ]
            except (TypeError, ValueError):
                continue
    return boxes


def import_capability(
    source,
    source_path,
    output,
    image_root,
    split,
    license_note,
    classes_path=None,
    groups_path=None,
) -> dict:
    """Normalize captionless capability datasets into CapabilityExample manifests."""
    if source not in CAPABILITY_SOURCES:
        raise ValueError(f"Unsupported capability source: {source}")
    rows = _load_capability_rows(source, Path(source_path), classes_path)
    output = Path(output)
    root = Path(image_root).resolve()
    group_of = _groups_or_default(groups_path, rows)
    records = []
    for index, row in enumerate(rows):
        image_rel, image_sha = _resolve_image(root, row["image"], output.parent)
        group = group_of(index, row)
        records.append(
            CapabilityExample(
                # Content-derived ID: split files imported separately cannot
                # collide on row indices; double imports are flagged as duplicates.
                sample_id=f"{source}-{sha(image_rel + json.dumps(row['capability'], sort_keys=True))[:16]}",
                image=image_rel,
                image_sha256=image_sha,
                dataset=source,
                split=split,
                event_id=str(group["event_id"]),
                source_group=str(group["source_group"]),
                license=license_note,
                source_url=group.get("source_url"),
                data_kind="research",
                extensions={"release_row": index, "capability": row["capability"]},
            )
        )
    result = _write_manifest(records, output)
    result["manifest_kind"] = "capability"
    return result


def _load_capability_rows(source, source_file, classes_path=None):
    if source == "sa1b":
        files = sorted(source_file.glob("*.json")) if source_file.is_dir() else [source_file]
        rows = []
        for path in files[:500_000]:
            data = strict_json(path.read_bytes())
            image_meta = data.get("image", {})
            width, height = image_meta.get("width") or 1, image_meta.get("height") or 1
            masks = []
            for annotation in data.get("annotations", [])[:200]:
                bbox = annotation.get("bbox")
                if not bbox:
                    continue
                x, y, w, h = (float(v) for v in bbox)
                masks.append(
                    {
                        "bbox": [x / width, y / height, (x + w) / width, (y + h) / height],
                        "area": annotation.get("area"),
                        "mask_id": annotation.get("id"),
                    }
                )
            rows.append(
                {
                    "image": image_meta.get("file_name") or Path(str(data.get("image_id", ""))).name,
                    "capability": {
                        "kind": "segmentation",
                        "mask_count": len(data.get("annotations", [])),
                        "masks": masks[:200],
                        "image_size": [width, height],
                    },
                }
            )
        return rows
    if source == "fsc-147":
        data = strict_json(source_file.read_bytes())
        classes = {}
        if classes_path:
            for line in Path(classes_path).read_text().splitlines():
                if "\t" in line:
                    name, label = line.split("\t", 1)
                    classes[name.strip()] = label.strip()
        rows = []
        for name, annotation in data.items():
            box = annotation.get("box") or [0, 0, 0, 0]
            rows.append(
                {
                    "image": name,
                    "capability": {
                        "kind": "counting",
                        "number": annotation.get("number"),
                        "class": classes.get(name),
                        "exemplar_box": box,
                        "caption_template": f"{annotation.get('number')} {classes.get(name, '')}".strip(),
                    },
                }
            )
        return rows
    if source in ("im2gps3k", "osv5m"):
        rows = []
        if source == "im2gps3k":
            for index, line in enumerate(source_file.read_text().splitlines()):
                parts = line.split()
                if len(parts) < 3:
                    continue
                url, lat, lon = parts[0], float(parts[1]), float(parts[2])
                rows.append(
                    {
                        "image": f"{index}.jpg",
                        "capability": {
                            "kind": "geolocation",
                            "gps": [lat, lon],
                            "source_url": url,
                        },
                    }
                )
        else:
            with source_file.open() as handle:
                for row in csv.DictReader(handle):
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                    rows.append(
                        {
                            "image": row["image"],
                            "capability": {
                                "kind": "geolocation",
                                "gps": [lat, lon],
                                "city": row.get("city"),
                                "country": row.get("country"),
                            },
                        }
                    )
        return rows
    if source == "vggface2":
        rows = []
        with source_file.open() as handle:
            for row in csv.reader(handle):
                if len(row) < 2:
                    continue
                class_id, name = row[0].strip(), row[1].strip()
                rows.append(
                    {
                        "image": f"{class_id}/{name}",
                        "capability": {
                            "kind": "identity",
                            "identity_class": class_id,
                            # Identity claims stay Unobservable without explicit evidence;
                            # this payload is for identity-verification research only.
                            "usage_note": "identity verification research only; consent/license required",
                        },
                    }
                )
        return rows
    if source == "yfcc100m":
        # YFCC100M metadata TSV positional order (official documentation):
        # photoid, userid, datetaken, dateuploaded, capturedevice, title,
        # description, usertags, machinetags, longitude, latitude, accuracy,
        # media, url
        columns = [
            "photoid",
            "userid",
            "datetaken",
            "dateuploaded",
            "capturedevice",
            "title",
            "description",
            "usertags",
            "machinetags",
            "longitude",
            "latitude",
            "accuracy",
            "media",
            "url",
        ]
        rows = []
        for line in source_file.read_text().splitlines():
            fields = line.split("\t")
            if len(fields) < len(columns):
                continue
            record = dict(zip(columns, fields, strict=True))
            try:
                lat = float(record["latitude"]) if record["latitude"] else None
                lon = float(record["longitude"]) if record["longitude"] else None
            except ValueError:
                lat = lon = None
            rows.append(
                {
                    "image": f"{record['photoid']}.jpg",
                    "capability": {
                        "kind": "photo-date-validation",
                        "capture_time": record["datetaken"] or None,
                        "gps": [lat, lon] if lat is not None and lon is not None else None,
                        "capture_device": record["capturedevice"] or None,
                        "photoid": record["photoid"],
                    },
                }
            )
        return rows
    raise ValueError(f"Unsupported source: {source}")


def load_capability_manifest(path):
    """Validate a capability manifest: unique IDs, split-safe grouping, bounded payload."""
    path = Path(path)
    records = [
        CapabilityExample.model_validate(strict_json(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    seen, groups = set(), {}
    for record in records:
        if record.sample_id in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(record.sample_id)
        for key in (
            ("image", record.image_sha256),
            ("event", record.event_id),
            ("source", record.source_group),
        ):
            if key in groups and groups[key] != record.split:
                raise ValueError(f"Split leakage: {key[0]} appears in multiple partitions")
            groups[key] = record.split
        if not record.license or not record.extensions.get("capability"):
            raise ValueError("License provenance and capability payload required")
    return records
