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


def grid_regions(image, asset_id, run_id, top_k=16):
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
                    region_id=f"rg_{run_id.replace('-', '')}_{len(result) + 1:06d}",
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


class OpenCLIPBackbone:
    def __init__(self):
        import open_clip

        pretrained = os.environ.get("AURORA_OPENCLIP_PRETRAINED", "")
        if not pretrained:
            raise ValueError("OPENCLIP_UNCONFIGURED: checkpoint belum dikonfigurasi")
        self.model_name = os.environ.get("AURORA_OPENCLIP_MODEL", "ViT-B-32")
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
):
    regions, crops = grid_regions(image, media.asset_id, run_id, top_k)
    descriptions = [a.statement for a in atoms] + [
        caption if caption is not None else " ".join(a.statement for a in atoms)
    ]
    config = {
        "backbone": backbone,
        "preprocessing": "exif-rgb-covering-grid-global-caption-v3",
        "top_k": top_k,
        "language": language,
        "library_version": version("open-clip-torch") if backbone == "openclip" else "local-color-v1",
    }
    configured = os.getenv("AURORA_OPENCLIP_PRETRAINED", "") if backbone == "openclip" else ""
    if backbone == "openclip":
        config.update(model=os.getenv("AURORA_OPENCLIP_MODEL", "ViT-B-32"), pretrained=configured)
    if configured and Path(configured).is_file():
        config["checkpoint_sha256"] = sha(Path(configured).read_bytes())
    key = sha(canonical({"image": media.sha256, "texts": descriptions, "config": config}))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    if path.is_file():
        data = np.load(path, allow_pickle=False)
        visual, text = torch.from_numpy(data["visual"].copy()), torch.from_numpy(data["text"].copy())
        cache_hit = True
    else:
        if backbone == "openclip":
            encoder = OpenCLIPBackbone()
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
