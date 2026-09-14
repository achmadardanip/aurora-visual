"""Frozen torchvision Mask R-CNN instance segmentation (pretrained, COCO).

Pretrained model for object/region proposals and instance counting; it is a
proposal/observation source only and never decides claim truth. The checkpoint
is downloaded explicitly (``aurora models --download segmentation``) into the
data directory; nothing is auto-downloaded at inference time.
"""

from pathlib import Path

MODEL_ID = "maskrcnn_resnet50_fpn_coco"
WEIGHTS_FILENAME = "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
WEIGHTS_URL = "https://download.pytorch.org/models/" + WEIGHTS_FILENAME
LICENSE = (
    "torchvision BSD-3-Clause; weights trained on COCO (annotations CC-BY 4.0; "
    "image licenses remain with original owners)"
)
_MIN_SCORE = 0.5


def weights_path(settings=None) -> Path:
    if settings is not None and settings.segmentation_weights:
        return Path(settings.segmentation_weights)
    if settings is not None:
        return settings.data_dir / "models" / "segmentation" / WEIGHTS_FILENAME
    import os

    return Path(os.getenv("AURORA_DATA_DIR", "var")) / "models" / "segmentation" / WEIGHTS_FILENAME


def weights_available(settings=None) -> bool:
    path = weights_path(settings)
    return path.is_file() and path.stat().st_size > 1_000_000


_CACHE: dict[str, "InstanceSegmenter"] = {}


class InstanceSegmenter:
    """Lazy frozen Mask R-CNN; boxes are normalized to [0,1] for the contract.

    The loaded model is cached per weights path: one server keeps at most one
    copy of the frozen checkpoint in memory across analyses.
    """

    @classmethod
    def load(cls, settings=None):
        path = str(weights_path(settings))
        if path not in _CACHE:
            _CACHE[path] = cls(settings)
        return _CACHE[path]

    def __init__(self, settings=None):
        import torch
        import torchvision
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

        path = weights_path(settings)
        if not weights_available(settings):
            raise ValueError("SEGMENTATION_UNCONFIGURED: bobot Mask R-CNN belum diunduh")
        self.weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        self.categories = list(self.weights.meta["categories"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device).requires_grad_(False)
        self.transforms = self.weights.transforms()

    def propose(self, image, top_k=32, min_score=_MIN_SCORE):
        """Return [{'label', 'score', 'bbox'}] sorted by score for one PIL image."""
        import torch

        tensor = self.transforms(image.convert("RGB")).to(self.device)
        with torch.inference_mode():
            output = self.model([tensor])[0]
        width, height = image.width, image.height
        detections = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"], strict=True):
            value = float(score)
            if value < min_score or int(label) >= len(self.categories):
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            x1, x2 = sorted((max(0.0, x1 / width), min(1.0, x2 / width)))
            y1, y2 = sorted((max(0.0, y1 / height), min(1.0, y2 / height)))
            if x1 >= x2 or y1 >= y2:
                continue
            detections.append(
                {
                    "label": self.categories[int(label)],
                    "score": value,
                    "bbox": [x1, y1, x2, y2],
                }
            )
        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections[: int(top_k)]

    def count(self, image, label=None, min_score=_MIN_SCORE):
        detections = self.propose(image, top_k=256, min_score=min_score)
        if label:
            detections = [item for item in detections if item["label"] == label]
        return detections
