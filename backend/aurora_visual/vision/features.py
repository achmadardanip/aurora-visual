import os
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from app.models.contract import Region, canonical, sha

COLOR_NAMES = ["merah", "biru", "hijau", "kuning", "hitam", "putih"]
ALIASES = {
    "red": "merah",
    "blue": "biru",
    "green": "hijau",
    "yellow": "kuning",
    "black": "hitam",
    "white": "putih",
}


def device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def feature_identity(config):
    """Language can vary within a bilingual corpus; encoder and grid settings cannot."""
    return {key: value for key, value in config.items() if key != "language"}


def color_descriptor(image):
    pixels = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    masks = [
        (r > 0.5) & (r > g * 1.5) & (r > b * 1.5),
        (b > 0.5) & (b > r * 1.5) & (b > g * 1.25),
        (g > 0.35) & (g > r * 1.4) & (g > b * 1.25),
        (r > 0.6) & (g > 0.5) & (b < 0.4),
        pixels.max(-1) < 0.18,
        pixels.min(-1) > 0.88,
    ]
    hist = np.array([mask.mean() for mask in masks], dtype=np.float32)
    return np.concatenate([hist, pixels.mean((0, 1)), pixels.std((0, 1))]).astype(np.float32)


def text_descriptor(text):
    tokens = set(re.findall(r"\w+", text.lower()))
    tokens |= {ALIASES[t] for t in list(tokens) if t in ALIASES}
    result = np.zeros(12, dtype=np.float32)
    for i, name in enumerate(COLOR_NAMES):
        result[i] = float(name in tokens)
    return result


def grid_regions(image, asset_id, run_id, top_k=16, start=0):
    if not 1 <= top_k <= 32:
        raise ValueError("top_k must be between 1 and 32")
    top_k = min(top_k, image.width * image.height)
    cols = min(image.width, int(np.ceil(np.sqrt(top_k))))
    rows = int(np.ceil(top_k / cols))
    if rows > image.height:
        rows = image.height
        cols = int(np.ceil(top_k / rows))
    result, crops = [], []
    for y in range(rows):
        count = min(cols, top_k - y * cols)
        for x in range(count):
            box = (x / count, y / rows, (x + 1) / count, (y + 1) / rows)
            result.append(
                Region(
                    region_id=f"rg_{run_id.replace('-', '')}_{start + len(result) + 1:06d}",
                    asset_id=asset_id,
                    bbox=box,
                    score=None,
                    description=f"Grid {y + 1},{x + 1}; bukan segmentasi objek",
                )
            )
            crops.append(
                image.crop(
                    tuple(round(v * (image.width if k % 2 == 0 else image.height)) for k, v in enumerate(box))
                )
            )
    return result, crops


def segmentation_regions(image, asset_id, run_id, top_k=16, start=0, proposer=None, settings=None):
    """Pretrained Mask R-CNN proposals replacing the grid; never a semantic claim."""
    if proposer is None:
        from aurora_visual.vision.segmentation import InstanceSegmenter

        proposer = InstanceSegmenter.load(settings)
    detections = proposer.propose(image, top_k=max(1, min(32, top_k)))
    if not detections:
        # No confident instance: fall back to the covering grid rather than an
        # empty region set (absence of detection is not evidence).
        return grid_regions(image, asset_id, run_id, top_k, start)
    result, crops = [], []
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        box = (x1, y1, x2, y2)
        label = str(detection.get("label", "object"))[:100]
        score = detection.get("score")
        result.append(
            Region(
                region_id=f"rg_{run_id.replace('-', '')}_{start + len(result) + 1:06d}",
                asset_id=asset_id,
                bbox=box,
                score=float(score) if score is not None else None,
                description=f"Mask R-CNN: {label} ({float(score):.2f}); proposal, bukan klaim".replace(
                    " (nan)", ""
                )[:500]
                if score is not None
                else f"Mask R-CNN: {label}; proposal, bukan klaim",
            )
        )
        crops.append(
            image.crop(
                (
                    round(x1 * image.width),
                    round(y1 * image.height),
                    round(x2 * image.width),
                    round(y2 * image.height),
                )
            )
        )
    return result, crops


def openclip_config(model=None, pretrained=None):
    """Resolve OpenCLIP config from explicit runtime settings, falling back to env."""
    model = model if model is not None else os.getenv("AURORA_OPENCLIP_MODEL", "ViT-B-32")
    pretrained = pretrained if pretrained is not None else os.getenv("AURORA_OPENCLIP_PRETRAINED", "")
    return model, pretrained


class OpenCLIPBackbone:
    def __init__(self, model_name=None, pretrained=None):
        import open_clip

        self.model_name, pretrained = openclip_config(model_name, pretrained)
        if not pretrained:
            raise ValueError("OPENCLIP_UNCONFIGURED: checkpoint belum dikonfigurasi")
        self.device = device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.version = f"openclip/{open_clip.__version__}/{self.model_name}/{pretrained}"

    def encode(self, crops, texts, language):
        # Standard OpenCLIP is English-centric; never claim Indonesian model validation.
        with torch.inference_mode():
            images = torch.stack([self.preprocess(crop) for crop in crops]).to(self.device)
            tokens = self.tokenizer(texts).to(self.device)
            features = self.model.encode_image(images, normalize=True).float().cpu()
            text = self.model.encode_text(tokens, normalize=True).float().cpu()
        return features, text


def extract(
    image,
    media,
    atoms,
    run_id,
    cache_dir: Path,
    backbone="local-color-v1",
    top_k=16,
    language="id",
    caption=None,
    openclip_model=None,
    openclip_pretrained=None,
    region_method="grid",
    region_proposer=None,
):
    if region_method == "segmentation":
        regions, crops = segmentation_regions(image, media.asset_id, run_id, top_k, 0, region_proposer)
    elif region_method == "grid":
        regions, crops = grid_regions(image, media.asset_id, run_id, top_k)
    else:
        raise ValueError("Unknown region method")
    descriptions = [a.statement for a in atoms] + [
        caption if caption is not None else " ".join(a.statement for a in atoms)
    ]
    config = {
        "backbone": backbone,
        "preprocessing": (
            "exif-rgb-covering-grid-global-caption-v3"
            if region_method == "grid"
            else "exif-rgb-covering-maskrcnn-global-caption-v3"
        ),
        "top_k": top_k,
        "language": language,
        "library_version": version("open-clip-torch") if backbone == "openclip" else "local-color-v1",
    }
    model_name, configured = ("", "")
    if backbone == "openclip":
        model_name, configured = openclip_config(openclip_model, openclip_pretrained)
        config.update(model=model_name, pretrained=configured)
    if configured and Path(configured).is_file():
        config["checkpoint_sha256"] = sha(Path(configured).read_bytes())
    key = sha(canonical({"images": [media.sha256], "texts": descriptions, "config": config}))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    if path.is_file():
        data = np.load(path, allow_pickle=False)
        visual, text = torch.from_numpy(data["visual"].copy()), torch.from_numpy(data["text"].copy())
        cache_hit = True
    else:
        if backbone == "openclip":
            encoder = OpenCLIPBackbone(model_name, configured)
            visual, text = encoder.encode([image, *crops], descriptions, language)
        elif backbone == "local-color-v1":
            visual = torch.from_numpy(np.stack([color_descriptor(crop) for crop in [image, *crops]]))
            text = torch.from_numpy(np.stack([text_descriptor(t) for t in descriptions]))
        else:
            raise ValueError("Unknown backbone")
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, visual=visual.numpy(), text=text.numpy())
        tmp.replace(path)
        cache_hit = False
    if not torch.isfinite(visual).all() or not torch.isfinite(text).all():
        raise ValueError("Non-finite embedding")
    return {
        "regions": regions,
        "visual": visual[1:],
        "text": text[:-1],
        "global_text": text[-1],
        "global": visual[0],
        "color": np.stack([color_descriptor(c) for c in crops]),
        "cache_key": key,
        "cache_hit": cache_hit,
        "config": config,
    }


def extract_multi(
    images,
    medias,
    atoms,
    run_id,
    cache_dir: Path,
    backbone="local-color-v1",
    top_k=16,
    language="id",
    caption=None,
    openclip_model=None,
    openclip_pretrained=None,
    region_method="grid",
    region_proposer=None,
    region_settings=None,
):
    """Extract merged features for several images: regions and crops from every asset.

    Region numbering is continuous across assets so region_id stays unique per run.
    Cached ``visual`` rows are grouped per image as [global, *crops]; the merged
    global feature is the mean over per-image global descriptors.
    """
    all_regions, groups = [], []
    for image, media in zip(images, medias):
        if region_method == "segmentation":
            regions, crops = segmentation_regions(
                image, media.asset_id, run_id, top_k, len(all_regions), region_proposer, region_settings
            )
        elif region_method == "grid":
            regions, crops = grid_regions(image, media.asset_id, run_id, top_k, start=len(all_regions))
        else:
            raise ValueError("Unknown region method")
        all_regions.extend(regions)
        groups.append((image, crops))
    descriptions = [a.statement for a in atoms] + [
        caption if caption is not None else " ".join(a.statement for a in atoms)
    ]
    config = {
        "backbone": backbone,
        "preprocessing": (
            "exif-rgb-covering-grid-global-caption-v3"
            if region_method == "grid"
            else "exif-rgb-covering-maskrcnn-global-caption-v3"
        ),
        "top_k": top_k,
        "language": language,
        "library_version": version("open-clip-torch") if backbone == "openclip" else "local-color-v1",
    }
    model_name, configured = ("", "")
    if backbone == "openclip":
        model_name, configured = openclip_config(openclip_model, openclip_pretrained)
        config.update(model=model_name, pretrained=configured)
    if configured and Path(configured).is_file():
        config["checkpoint_sha256"] = sha(Path(configured).read_bytes())
    key = sha(
        canonical({"images": sorted(m.sha256 for m in medias), "texts": descriptions, "config": config})
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    if path.is_file():
        data = np.load(path, allow_pickle=False)
        visual, text = torch.from_numpy(data["visual"].copy()), torch.from_numpy(data["text"].copy())
        cache_hit = True
    else:
        if backbone == "openclip":
            encoder = OpenCLIPBackbone(model_name, configured)
            flat = [item for image, crops in groups for item in [image, *crops]]
            visual, text = encoder.encode(flat, descriptions, language)
        elif backbone == "local-color-v1":
            rows = [color_descriptor(item) for image, crops in groups for item in [image, *crops]]
            visual = torch.from_numpy(np.stack(rows))
            text = torch.from_numpy(np.stack([text_descriptor(t) for t in descriptions]))
        else:
            raise ValueError("Unknown backbone")
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, visual=visual.numpy(), text=text.numpy())
        tmp.replace(path)
        cache_hit = False
    if not torch.isfinite(visual).all() or not torch.isfinite(text).all():
        raise ValueError("Non-finite embedding")
    region_rows, global_rows, offset = [], [], 0
    for image, crops in groups:
        size = 1 + len(crops)
        global_rows.append(visual[offset])
        region_rows.append(visual[offset + 1 : offset + size])
        offset += size
    regional = torch.cat(region_rows, dim=0)
    return {
        "regions": all_regions,
        "visual": regional,
        "text": text[:-1],
        "global_text": text[-1],
        "global": torch.stack(global_rows).mean(0),
        "color": regional.numpy(),
        "cache_key": key,
        "cache_hit": cache_hit,
        "config": config,
    }


def model_inputs(atoms, features):
    from aurora_visual.alignment.model import ROLES

    meta = []
    for a in atoms:
        role = [float(a.role == r) for r in ROLES]
        meta.append(
            role
            + [
                float(a.qualifiers.negated),
                min(a.qualifiers.quantity or 0, 100) / 100,
                float(a.qualifiers.time is not None or a.qualifiers.location is not None),
                a.parser_confidence or 0.0,
            ]
        )
    geometry = torch.tensor(
        [[*r.bbox, r.score if r.score is not None else 0] for r in features["regions"]], dtype=torch.float32
    )
    # First-pass frozen anchors condition a unary relative-geometry cost, not graph OT.
    from torch.nn import functional as F

    prior = (F.normalize(features["text"], dim=-1) @ F.normalize(features["visual"], dim=-1).T / 0.1).softmax(
        -1
    )
    centers = (geometry[:, :2] + geometry[:, 2:4]) / 2
    anchors = prior @ centers
    indexes = {a.atom_id: i for i, a in enumerate(atoms)}
    targets, masks = [], []
    for atom in atoms:
        dependencies = [indexes[d] for d in atom.depends_on if d in indexes]
        target = anchors[dependencies].mean(0) if dependencies else torch.tensor([0.5, 0.5])
        statement = atom.statement.lower()
        offset = torch.tensor(
            [
                -0.25
                if "kiri" in statement or "left of" in statement
                else 0.25
                if "kanan" in statement or "right of" in statement
                else 0.0,
                -0.25
                if "di atas" in statement or "above" in statement
                else 0.25
                if "di bawah" in statement or "below" in statement
                else 0.0,
            ]
        )
        targets.append((target + offset).clamp(0, 1))
        masks.append(float(bool(dependencies) and atom.role == "relation"))
    return dict(
        text=features["text"],
        regions=features["visual"],
        atom_meta=torch.tensor(meta),
        geometry=geometry,
        global_feature=features["global"],
        global_text=features["global_text"],
        relation_targets=torch.stack(targets),
        relation_mask=torch.tensor(masks),
    )
