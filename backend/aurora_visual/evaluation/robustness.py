from PIL import ImageDraw, ImageFilter


def perturb_image(image, kind):
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2)), {"coordinate_transform": "identity"}
    if kind == "crop":
        left, top, right, bottom = 0.15, 0.15, 0.85, 0.85
        crop = image.crop(
            (
                round(left * image.width),
                round(top * image.height),
                round(right * image.width),
                round(bottom * image.height),
            )
        )
        return crop, {
            "coordinate_transform": [left, top, right, bottom],
            "warning": "crop reduces coverage; never create count contradiction",
        }
    if kind == "ocr-overlay":
        result = image.copy()
        draw = ImageDraw.Draw(result)
        draw.rectangle((0, 0, image.width, max(22, image.height // 8)), fill="white")
        draw.text((5, 5), "SEPTEMBER 2026", fill="black")
        return result, {"coordinate_transform": "identity", "warning": "overlay text is not capture date"}
    raise ValueError("Unknown robustness perturbation")


def evaluate_pixels(
    model,
    image,
    caption,
    language,
    cache_dir,
    backbone="local-color-v1",
    method="uot",
    *,
    checkpoint_metadata,
    allow_fixture_mismatch=False,
):
    """Measure prediction stability; transformed images do not inherit gold labels."""
    import io
    from uuid import uuid4

    import torch
    from app.models.contract import MediaRef, sha

    from aurora_visual.atomization.parser import RuleAtomizer
    from aurora_visual.training.engine import check_feature_compatibility
    from aurora_visual.vision.features import extract, feature_identity, model_inputs

    atoms = RuleAtomizer().parse(caption, language).atoms
    results = []
    for kind in ("original", "blur", "crop", "ocr-overlay"):
        altered, transform = (
            (image, {"coordinate_transform": "identity"})
            if kind == "original"
            else perturb_image(image, kind)
        )
        stream = io.BytesIO()
        altered.save(stream, "PNG")
        digest = sha(stream.getvalue())
        ref = MediaRef(
            asset_id="asset_" + digest,
            sha256=digest,
            media_type="image/png",
            width=altered.width,
            height=altered.height,
            uri="dataset/perturbation",
        )
        features = extract(altered, ref, atoms, str(uuid4()), cache_dir, backbone, 16, language, caption)
        compatibility = "matched"
        try:
            check_feature_compatibility(
                checkpoint_metadata, [{"feature_config": feature_identity(features["config"])}]
            )
        except ValueError:
            if not allow_fixture_mismatch or checkpoint_metadata["data_kind"] != "fixture":
                raise
            # Explicit smoke-only wiring exercise across synthetic and pixel descriptors.
            # A research checkpoint must always match its encoder/preprocessing metadata.
            compatibility = "fixture_override_unvalidated"
        with torch.inference_mode():
            output = model(**model_inputs(atoms, features), method=method)
        results.append(
            {
                "kind": kind,
                "feature_compatibility": compatibility,
                "transform": transform,
                "predictions": output["probabilities"].argmax(-1).tolist(),
                "probabilities": output["probabilities"].tolist(),
                "unmatched_mass": output["transport"].unmatched_mass.tolist(),
            }
        )
    baseline = results[0]["predictions"]
    for result in results:
        result["prediction_agreement"] = sum(
            a == b for a, b in zip(baseline, result["predictions"], strict=True)
        ) / len(baseline)
    return {
        "atoms": [a.model_dump(mode="json") for a in atoms],
        "variants": results,
        "gold_robustness_metrics": None,
        "method": method,
        "note": "Stability is not correctness; crop/overlay gold requires separate human review. Fixture feature overrides verify wiring only, not model robustness.",
    }
