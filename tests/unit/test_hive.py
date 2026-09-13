import copy
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from app.models.contract import Atom, MediaRef, Qualifiers, Span, VisualAssessment
from aurora_visual.atomization.parser import HiveVLMAtomizer, validate_structured_atoms
from aurora_visual.hive import (
    _MAX_V2_RESPONSE,
    HiveError,
    HiveV2Client,
    HiveV3Client,
    fuse_observations,
    group_v2_capabilities,
    normalize_observations,
    normalize_translation,
    normalize_v2,
    ocr_from_models,
    regions_for_detections,
    select_models,
    stage1_from_v2,
    v2_capability_status,
)


def atom(
    atom_id="a000001",
    role="object",
    statement="A membawa tas",
    subject="A",
    predicate="membawa",
    obj="tas",
    start=0,
    end=13,
    depends_on=None,
    negated=False,
):
    return Atom(
        atom_id=atom_id,
        statement=statement,
        role=role,
        subject=subject,
        predicate=predicate,
        object=obj,
        qualifiers=Qualifiers(negated=negated, quantity=None, time=None, location=None),
        spans=[Span(start=start, end=end)],
        depends_on=depends_on or [],
        check_worthiness=1.0,
        parser_confidence=None,
    )


def successful_entry(model, model_type, output):
    return {
        "status": {"code": "0", "message": "SUCCESS"},
        "response": {
            "input": {
                "model": model,
                "model_type": model_type,
                "model_version": "fixture-1",
            },
            "output": output,
        },
    }


def failed_entry(model="failed-model", code="MODEL_DISABLED"):
    return {
        "status": {"code": code, "message": "FAILED"},
        "response": {"input": {"model": model, "model_type": "fixture"}},
    }


def v2_payload(*entries):
    return {"status": list(entries)}


def test_v2_media_request_uses_token_and_media_without_leaking_secret(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"original-bytes")
    seen = {}

    def handler(request):
        seen["authorization"] = request.headers["authorization"]
        seen["content_type"] = request.headers["content-type"]
        seen["body"] = request.read()
        return httpx.Response(200, json=v2_payload(successful_entry("ocr", "ocr", [])))

    client = HiveV2Client("server-only-secret", transport=httpx.MockTransport(handler))
    result = client.submit_media(image)
    assert result["payload"]["status"][0]["status"]["code"] == "0"
    assert seen["authorization"] == "Token server-only-secret"
    assert "multipart/form-data" in seen["content_type"]
    assert b'name="media"' in seen["body"] and b"original-bytes" in seen["body"]
    assert "server-only-secret" not in repr(result)


def test_v2_translation_uses_exact_text_data_and_language_options():
    seen = {}

    def handler(request):
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = request.read().decode()
        payload = v2_payload(
            successful_entry("translation", "translation", [{"translated_text": "A carries a bag"}])
        )
        return httpx.Response(200, json=payload)

    client = HiveV2Client("translation-secret", transport=httpx.MockTransport(handler))
    result = client.translate("A membawa tas", "id", "en")
    assert seen["authorization"] == "Token translation-secret"
    assert "text_data=A+membawa+tas" in seen["body"]
    assert "%22input_language%22%3A%22ID%22" in seen["body"]
    assert "%22output_language%22%3A%22EN%22" in seen["body"]
    assert normalize_translation(result) == "A carries a bag"
    with pytest.raises(ValueError, match="1–512"):
        client.translate("x" * 513, "id", "en")


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (httpx.Response(302, headers={"location": "https://example.invalid"}), "redirect_rejected"),
        (httpx.Response(429, json={"error": "limited"}), "rate_limited"),
        (httpx.Response(503, text="private provider body"), "http_error"),
        (
            httpx.Response(200, headers={"content-length": str(_MAX_V2_RESPONSE + 1)}),
            "response_too_large",
        ),
        (httpx.Response(200, content=b'{"x":1,"x":2}'), "malformed_response"),
        (httpx.Response(200, content=b'{"x":NaN}'), "malformed_response"),
    ],
)
def test_v2_errors_are_bounded_and_redacted(tmp_path, response, code):
    image = tmp_path / "sample.png"
    image.write_bytes(b"safe")
    client = HiveV2Client("top-secret", transport=httpx.MockTransport(lambda _: response))
    with pytest.raises(HiveError) as raised:
        client.submit_media(image)
    assert raised.value.code == code
    assert "top-secret" not in str(raised.value)
    assert "private provider body" not in str(raised.value)


def test_v2_timeout_and_network_errors_are_sanitized(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"safe")
    for exception, expected in [
        (httpx.ReadTimeout("provider URL and secret"), "timeout"),
        (httpx.ConnectError("provider URL and secret"), "network_error"),
    ]:
        client = HiveV2Client(
            "secret-value",
            transport=httpx.MockTransport(lambda _: (_ for _ in ()).throw(exception)),
        )
        with pytest.raises(HiveError) as raised:
            client.submit_media(image)
        assert raised.value.code == expected
        assert "secret-value" not in str(raised.value)
        assert "provider URL" not in str(raised.value)


def test_v2_streamed_response_limit(tmp_path):
    class OversizeStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"x" * _MAX_V2_RESPONSE
            yield b"x"

    image = tmp_path / "sample.png"
    image.write_bytes(b"safe")
    response = httpx.Response(200, stream=OversizeStream())
    client = HiveV2Client("secret", transport=httpx.MockTransport(lambda _: response))
    with pytest.raises(HiveError, match="response_too_large"):
        client.submit_media(image)


def test_mixed_v2_entries_preserve_success_and_failure():
    payload = v2_payload(
        successful_entry(
            "common-object-detection",
            "classification",
            [
                {
                    "bounding_poly": [
                        {
                            "dimensions": {"left": 10, "top": 5, "right": 50, "bottom": 25},
                            "classes": [{"class": "bag", "score": 0.91}],
                            "meta": {"type": "object"},
                        }
                    ]
                }
            ],
        ),
        failed_entry(code="PROJECT_MODEL_DISABLED"),
    )
    normalized = normalize_v2(payload, 100, 50)
    assert len(normalized["models"]) == 1
    assert normalized["models"][0]["detections"][0]["bbox"] == [0.1, 0.1, 0.5, 0.5]
    assert normalized["entries"][1] == {
        "model": "failed-model",
        "model_type": "fixture",
        "status": "failed",
        "error_code": "PROJECT_MODEL_DISABLED",
    }
    assert v2_capability_status(normalized, "object") == ("ok", None)
    assert v2_capability_status(normalized, "ocr") == ("failed", "PROJECT_MODEL_DISABLED")


def test_all_failed_v2_entries_are_explicit_failures():
    normalized = normalize_v2(v2_payload(failed_entry(code="RATE_LIMITED")), 100, 100)
    assert normalized["models"] == []
    assert v2_capability_status(normalized, "origin") == ("failed", "RATE_LIMITED")


def test_v2_origin_semantics_keep_aggregate_face_and_metadata_distinct():
    payload = v2_payload(
        successful_entry(
            "ai-generated-image-detection",
            "classification",
            [
                {
                    "classes": [
                        {"class": "ai_generated", "score": 0.96},
                        {"class": "Stable Diffusion", "score": 0.82},
                    ],
                    "algorithmic_tags": {
                        "c2pa": {"present": True},
                        "xmp": {"present": True},
                        "exif": {"present": False},
                    },
                }
            ],
        ),
        successful_entry(
            "deepfake-detection",
            "face-detection",
            [
                {
                    "classes": [{"class": "deepfake", "score": 0.94}],
                    "bounding_poly": [
                        {
                            "vertices": [[20, 10], [60, 10], [60, 50], [20, 50]],
                            "classes": [
                                {"class": "yes_deepfake", "score": 0.91},
                                {"class": "no_deepfake", "score": 0.09},
                            ],
                            "meta": {"type": "face"},
                        }
                    ],
                }
            ],
        ),
    )
    normalized = normalize_v2(payload, 100, 100)
    detectors, metadata = stage1_from_v2({"models": normalized["models"]})
    ai, face = detectors
    assert ai["assessment"] == "likely_ai_generated"
    assert ai["source_attribution"][0]["label"] == "Stable Diffusion"
    assert face["assessment"] == "likely_deepfake"
    assert face["faces"][0]["bbox"] == [0.2, 0.1, 0.6, 0.5]
    assert "seluruh gambar" in face["message"]
    assert metadata["verification"] == "provider_observation_not_cryptographically_verified"
    assert metadata["observations"][0]["c2pa"]["present"] is True


def test_not_ai_and_missing_deepfake_are_not_authenticity_evidence():
    normalized = normalize_v2(
        v2_payload(
            successful_entry(
                "ai-generated-image-detection",
                "classification",
                [{"classes": [{"class": "not_ai_generated", "score": 0.99}]}],
            )
        ),
        20,
        20,
    )
    ai, deepfake = stage1_from_v2({"models": normalized["models"]})[0]
    assert ai["status"] == "inconclusive" and ai["assessment"] == "uncertain"
    assert deepfake["status"] == "unsupported" and deepfake["assessment"] == "not_assessed"


def test_explicit_no_face_is_not_applicable_not_authenticity():
    normalized = normalize_v2(
        v2_payload(
            successful_entry(
                "deepfake-detection",
                "classification",
                [{"classes": [{"class": "no_face_detected", "score": 1.0}]}],
            )
        ),
        20,
        20,
    )
    deepfake = stage1_from_v2({"models": normalized["models"]})[0][1]
    assert deepfake["status"] == "not_applicable"
    assert deepfake["assessment"] == "no_face_observed"
    assert "bukan bukti" in deepfake["message"]


def test_stage3_models_normalize_provider_specific_diagnostics():
    payload = v2_payload(
        successful_entry(
            "ocr-text-recognition",
            "ocr",
            [
                {
                    "block_text": "TULISAN TERLIHAT",
                    "bounding_poly": [
                        {
                            "dimensions": {"left": 4, "top": 5, "right": 40, "bottom": 20},
                            "classes": [{"class": "TULISAN", "score": 0.93}],
                            "meta": {"type": "text", "score": 0.92, "text": "TULISAN"},
                        }
                    ],
                }
            ],
        ),
        successful_entry(
            "people-counting",
            "classification",
            [{"num_people": "5+"}, {"num_people": 3}, {"num_people": "6"}],
        ),
        successful_entry(
            "logo-recognition-location",
            "detection",
            [
                {
                    "bounding_poly": [
                        {
                            "vertices": [
                                {"x": 10, "y": 10},
                                {"x": 30, "y": 10},
                                {"x": 30, "y": 30},
                                {"x": 10, "y": 30},
                            ],
                            "classes": [{"class": "Example logo", "score": 0.88}],
                            "meta": {
                                "type": "logo",
                                "clarity": 0.7,
                                "locations": [{"country": "ID"}],
                            },
                        }
                    ]
                }
            ],
        ),
        successful_entry(
            "celebrity-recognition",
            "face-detection",
            [
                {
                    "bounding_poly": [
                        {
                            "dimensions": {"left": 50, "top": 20, "right": 90, "bottom": 80},
                            "classes": [{"class": "Example Person", "score": 0.74}],
                            "meta": {"type": "face"},
                        }
                    ]
                }
            ],
        ),
    )
    normalized = normalize_v2(payload, 100, 100)
    assert len(select_models(normalized, ["ocr"])) == 1
    assert len(select_models(normalized, ["people"])) == 1
    assert select_models(normalized, ["people"])[0]["people_counts"] == ["5+", "3"]
    logo = select_models(normalized, ["logo"])[0]["detections"][0]
    assert logo["type"] == "logo" and logo["locations"] == [{"country": "ID"}]
    celebrity = select_models(normalized, ["celebrity"])[0]["detections"][0]
    assert celebrity["classes"][0] == {"label": "Example Person", "score": 0.74}
    ocr, provenance = ocr_from_models(normalized["models"], "id")
    assert ocr[0].text == "TULISAN" and provenance[0]["provider"] == "hive-v2"


def test_provider_regions_preserve_asset_and_run_identity():
    run_id = str(uuid4())
    media = MediaRef(
        asset_id="asset_" + "a" * 64,
        sha256="a" * 64,
        media_type="image/png",
        width=100,
        height=100,
        uri="api/v1/media/asset_" + "a" * 64,
    )
    regions = regions_for_detections(
        [{"bbox": [0.1, 0.2, 0.4, 0.6], "classes": [{"label": "bag", "score": 0.9}], "type": "object"}],
        media,
        run_id,
    )
    assert regions[0].asset_id == media.asset_id
    assert regions[0].region_id.startswith("rg_" + run_id.replace("-", "") + "_")


def valid_structured_atom(caption="A membawa tas"):
    return {
        "atom_id": "a000001",
        "statement": caption,
        "role": "object",
        "subject": "A",
        "predicate": "membawa",
        "object": "tas",
        "qualifiers": {"negated": False, "quantity": None, "time": None, "location": None},
        "spans": [{"start": 0, "end": len(caption)}],
        "depends_on": [],
        "check_worthiness": 1.0,
        "parser_confidence": 0.9,
    }


def test_v3_uses_bearer_structured_output_and_nested_strict_json():
    seen = {}

    def handler(request):
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = json.loads(request.read())
        nested = json.dumps({"atoms": [valid_structured_atom()]})
        return httpx.Response(200, json={"choices": [{"message": {"content": nested}}]})

    client = HiveV3Client("organization-secret", transport=httpx.MockTransport(handler))
    result = client.parse_atoms("A membawa tas", "id")
    assert result["atoms"][0]["statement"] == "A membawa tas"
    assert seen["authorization"] == "Bearer organization-secret"
    assert seen["body"]["model"] == "hive/vision-language-model"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True
    assert "organization-secret" not in repr(result)


def test_v3_rejects_duplicate_outer_and_nested_json():
    for content in [
        b'{"choices":[],"choices":[]}',
        json.dumps({"choices": [{"message": {"content": '{"atoms":[],"atoms":[]}'}}]}).encode(),
    ]:
        client = HiveV3Client(
            "secret",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content)),
        )
        with pytest.raises(HiveError, match="malformed_response"):
            client.parse_atoms("A membawa tas", "id")


def test_hive_atomizer_preserves_caption_spans_and_nulls_confidence():
    caption = "📷 A membawa tas"
    raw = valid_structured_atom(caption)
    raw["subject"] = "A"
    raw["statement"] = "A membawa tas"
    raw["spans"] = [{"start": 2, "end": len(caption)}]

    class Client:
        def parse_atoms(self, submitted, language):
            assert submitted == caption and language == "id"
            return {"atoms": [raw]}

    result = HiveVLMAtomizer(Client()).parse(caption)
    assert result.atoms[0].parser_confidence is None
    assert caption[result.atoms[0].spans[0].start : result.atoms[0].spans[0].end] == "A membawa tas"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"object": "invented"}),
        lambda value: value.update({"depends_on": ["a999999"]}),
        lambda value: value.update({"spans": [{"start": 0, "end": 999}]}),
    ],
)
def test_structured_atoms_reject_ungrounded_dangling_and_invalid_spans(mutation):
    value = valid_structured_atom()
    mutation(value)
    with pytest.raises(ValueError):
        validate_structured_atoms({"atoms": [value]}, "A membawa tas")


def test_structured_atoms_reject_cycles():
    first = valid_structured_atom("A membawa tas")
    second = copy.deepcopy(first)
    first["depends_on"] = ["a000002"]
    second["atom_id"] = "a000002"
    second["depends_on"] = ["a000001"]
    with pytest.raises(ValueError, match="Cyclic"):
        validate_structured_atoms({"atoms": [first, second]}, "A membawa tas")


def test_structured_atoms_reject_duplicate_ids_dependencies_and_negation_mismatch():
    first = valid_structured_atom("A tidak membawa tas")
    second = copy.deepcopy(first)
    with pytest.raises(ValueError, match="Duplicate atom identifiers"):
        validate_structured_atoms({"atoms": [first, second]}, "A tidak membawa tas")

    second["atom_id"] = "a000002"
    second["depends_on"] = ["a000001", "a000001"]
    first["qualifiers"]["negated"] = True
    second["qualifiers"]["negated"] = True
    with pytest.raises(ValueError, match="Duplicate atom dependency"):
        validate_structured_atoms({"atoms": [first, second]}, "A tidak membawa tas")

    first = valid_structured_atom("A tidak membawa tas")
    with pytest.raises(ValueError, match="negation"):
        validate_structured_atoms({"atoms": [first]}, "A tidak membawa tas")


def test_v3_observation_uses_data_uri_and_enforces_encoded_limit(tmp_path, monkeypatch):
    image = tmp_path / "preview.png"
    image.write_bytes(b"preview-bytes")
    seen = {}

    def complete(messages, schema, name, max_tokens):
        seen["messages"] = messages
        return {"observations": []}

    client = HiveV3Client("secret")
    monkeypatch.setattr(client, "complete", complete)
    assert client.observe(image, "image/png", "A membawa tas", [atom()]) == {"observations": []}
    content = seen["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    monkeypatch.setattr("aurora_visual.hive._MAX_V3_DATA_URI_BYTES", 20)
    with pytest.raises(HiveError, match="media_too_large"):
        client.observe(image, "image/png", "A membawa tas", [atom()])


@pytest.mark.parametrize("count", ["0", "1", "2", "3", "4", "5+"])
def test_people_count_keeps_provider_categories(count):
    normalized = normalize_v2(
        v2_payload(successful_entry("people-counting", "classification", [{"num_people": count}])),
        10,
        10,
    )
    assert select_models(normalized, ["people"])[0]["people_counts"] == [count]


def test_grouping_requires_same_key_and_representation():
    settings = SimpleNamespace(
        hive_v2_key=lambda capability: {
            "origin": "shared",
            "ocr": "shared",
            "object": "shared",
            "scene": "scene-key",
            "people": "",
            "logo": "shared",
            "celebrity": "celebrity-key",
        }[capability]
    )
    assert group_v2_capabilities(settings) == {
        ("shared", "original"): ["origin"],
        ("shared", "preview"): ["ocr", "object", "logo"],
        ("scene-key", "preview"): ["scene"],
        ("celebrity-key", "preview"): ["celebrity"],
    }


def observation(**overrides):
    value = {
        "atom_id": "a000001",
        "visibility": "visible",
        "relation": "supports",
        "observation": "Sebuah tas terlihat",
        "alternative": None,
        "confidence": 0.91,
        "bbox": [0.1, 0.2, 0.6, 0.8],
    }
    value.update(overrides)
    return value


def assessment(status="Unobservable", **overrides):
    value = {
        "atom_id": "a000001",
        "visual_status": status,
        "probabilities": None,
        "unmatched_mass": None,
        "observability_score": None,
        "supporting_regions": [],
        "contradicting_regions": [],
        "counter_evidence": None,
        "rationale": "Belum ada bukti eksplisit.",
        "inference_kind": "heuristic",
    }
    value.update(overrides)
    return VisualAssessment(**value)


def test_observation_normalization_downgrades_incomplete_contradiction():
    normalized = normalize_observations(
        {"observations": [observation(relation="contradicts", alternative=None)]},
        {"a000001"},
    )
    assert normalized[0]["relation"] == "uncertain"


@pytest.mark.parametrize(
    "value,code",
    [
        ({"observations": [observation(), observation()]}, "invalid_atom_reference"),
        ({"observations": [observation(atom_id="a999999")]}, "invalid_atom_reference"),
        ({"observations": [observation(bbox=[0.6, 0.2, 0.1, 0.8])]}, "invalid_bbox"),
        ({"observations": [observation(bbox=[0.1, 0.2, float("nan"), 0.8])]}, "invalid_bbox"),
        ({"observations": [observation(confidence=float("nan"))]}, "malformed_response"),
        ({"observations": [observation(extra="not allowed")]}, "malformed_response"),
    ],
)
def test_observation_normalization_rejects_invalid_provider_data(value, code):
    with pytest.raises(HiveError, match=code):
        normalize_observations(value, {"a000001"})


def test_fusion_support_and_explicit_contradiction_require_region():
    media = MediaRef(
        asset_id="asset_" + "b" * 64,
        sha256="b" * 64,
        media_type="image/png",
        width=100,
        height=100,
        uri="api/v1/media/asset_" + "b" * 64,
    )
    run_id = str(uuid4())
    supported = fuse_observations(
        [assessment()],
        [observation()],
        [atom()],
        media,
        run_id,
    )[0]
    assert supported.visual_status == "Supported"
    assert supported.supporting_regions[0].asset_id == media.asset_id

    contradicted = fuse_observations(
        [assessment()],
        [observation(relation="contradicts", alternative="Tidak ada tas")],
        [atom()],
        media,
        run_id,
    )[0]
    assert contradicted.visual_status == "Contradicted"
    assert contradicted.counter_evidence == "Tidak ada tas"
    assert contradicted.contradicting_regions


@pytest.mark.parametrize(
    "candidate_atom,candidate_observation",
    [
        (atom(role="actor"), observation()),
        (atom(), observation(confidence=0.79)),
        (atom(), observation(bbox=None)),
        (atom(), observation(visibility="not_visible")),
        (atom(), observation(relation="uncertain")),
    ],
)
def test_fusion_preserves_unobservable_without_eligible_explicit_evidence(
    candidate_atom, candidate_observation
):
    media = MediaRef(
        asset_id="asset_" + "c" * 64,
        sha256="c" * 64,
        media_type="image/png",
        width=100,
        height=100,
        uri="api/v1/media/asset_" + "c" * 64,
    )
    result = fuse_observations(
        [assessment()],
        [candidate_observation],
        [candidate_atom],
        media,
        str(uuid4()),
    )[0]
    assert result.visual_status == "Unobservable"
    assert not result.supporting_regions and not result.contradicting_regions


def test_fusion_disagreement_clears_local_evidence():
    media = MediaRef(
        asset_id="asset_" + "d" * 64,
        sha256="d" * 64,
        media_type="image/png",
        width=100,
        height=100,
        uri="api/v1/media/asset_" + "d" * 64,
    )
    local_region = regions_for_detections(
        [{"bbox": [0.1, 0.1, 0.9, 0.9], "classes": [], "score": 0.9, "type": "grid"}],
        media,
        str(uuid4()),
    )[0]
    result = fuse_observations(
        [assessment("Supported", supporting_regions=[local_region], observability_score=0.9)],
        [observation(relation="contradicts", alternative="Tas tidak terlihat")],
        [atom()],
        media,
        str(uuid4()),
    )[0]
    assert result.visual_status == "Unobservable"
    assert result.observability_score is None
    assert not result.supporting_regions and not result.contradicting_regions
    assert result.counter_evidence is None
