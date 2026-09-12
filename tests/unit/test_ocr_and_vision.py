from uuid import uuid4

import pytest
from app.models.contract import MediaRef
from aurora_visual.alignment.uot import align
from aurora_visual.atomization.parser import RuleAtomizer
from aurora_visual.entailment.reasoning import assess
from aurora_visual.evaluation.robustness import perturb_image
from aurora_visual.ocr.engine import capability, recognize
from aurora_visual.vision.features import extract
from PIL import Image, ImageDraw, ImageFont


def test_real_ocr_is_observation_not_capture_date(tmp_path):
    if capability()["status"] != "ok":
        pytest.skip("Tesseract ind+eng unavailable")
    image = Image.new("RGB", (640, 160), "white")
    ImageDraw.Draw(image).text((20, 40), "SEPTEMBER 2026", fill="black", font=ImageFont.load_default(size=50))
    ocr, warnings = recognize(image)
    assert any("2026" in word.text for word in ocr)
    atoms = RuleAtomizer().parse("Peristiwa terjadi pada September 2026").atoms
    ref = MediaRef(
        asset_id="asset_" + "a" * 64,
        sha256="a" * 64,
        media_type="image/png",
        width=640,
        height=160,
        uri="dataset/image",
    )
    f = extract(image, ref, atoms, str(uuid4()), tmp_path)
    import torch

    results = assess(atoms, f, align(torch.ones(len(atoms), 16)), ocr)
    assert all(a.visual_status == "Unobservable" for a in results)
    assert any(w.code == "OCR_OBSERVATION_ONLY" for w in warnings)


def test_cache_invalidation_and_grid_cap(tmp_path):
    image = Image.new("RGB", (60, 40), "red")
    ref = MediaRef(
        asset_id="asset_" + "b" * 64,
        sha256="b" * 64,
        media_type="image/png",
        width=60,
        height=40,
        uri="dataset/image",
    )
    atoms = RuleAtomizer().parse("Bidang ini berwarna merah").atoms
    a = extract(image, ref, atoms, str(uuid4()), tmp_path)
    b = extract(image, ref, atoms, str(uuid4()), tmp_path)
    c = extract(image, ref, atoms, str(uuid4()), tmp_path, top_k=9)
    assert not a["cache_hit"] and b["cache_hit"] and c["cache_key"] != b["cache_key"]
    assert len(c["regions"]) == 9
    with pytest.raises(ValueError):
        extract(image, ref, atoms, str(uuid4()), tmp_path, top_k=33)


@pytest.mark.parametrize("kind", ["blur", "crop", "ocr-overlay"])
def test_robustness_transforms(kind):
    image = Image.new("RGB", (100, 100), "red")
    transformed, metadata = perturb_image(image, kind)
    assert transformed.width > 0 and metadata["coordinate_transform"]
    if kind == "crop":
        assert transformed.size == (70, 70)


def test_grid_covers_image_even_when_k_not_square():
    from aurora_visual.vision.features import grid_regions

    for size in ((80, 60), (1, 1), (1, 40), (40, 1)):
        for k in (1, 7, 16, 17, 31, 32):
            regions, crops = grid_regions(Image.new("RGB", size), "asset_" + "c" * 64, str(uuid4()), k)
            assert sum((r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]) for r in regions) == pytest.approx(
                1.0
            )
            assert all(crop.width > 0 and crop.height > 0 for crop in crops)
