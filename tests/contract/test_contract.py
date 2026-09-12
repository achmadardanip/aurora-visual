import copy
from pathlib import Path

import jsonschema
import pytest
from app.models.contract import AuroraBundle, atom_set_id, canonical, strict_json
from conftest import run
from pydantic import ValidationError


def test_schema_generated_and_bundle_validate(client, app, demo):
    result, _ = run(client, app, demo)
    b = AuroraBundle.model_validate(result)
    jsonschema.validate(result, AuroraBundle.model_json_schema())
    assert b.analysis.atom_set_id == atom_set_id(b, b.analysis.atomic_claims)
    assert b.retrieval is None and b.decision is None


def test_duplicate_keys_nan_infinity():
    for data in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e400}'):
        with pytest.raises(ValueError):
            strict_json(data)


def test_jcs_known_vector():
    assert (
        canonical({"z": -0.0, "a": 1e-7, "b": 1e20, "é": "📷"})
        == '{"a":1e-7,"b":100000000000000000000,"z":0,"é":"📷"}'.encode()
    )


def test_unsupported_version_empty_caption_and_key(client, demo):
    for changes, code in [({"schema_version": "2.0.0"}, "SCHEMA_VERSION_UNSUPPORTED")]:
        response = client.post("/api/v1/analyze", json={**demo, **changes}, headers={"Idempotency-Key": "x"})
        assert response.status_code == 422 and response.json()["error"]["code"] == code
    bad = copy.deepcopy(demo)
    bad["input"]["claim_text"] = " "
    assert client.post("/api/v1/analyze", json=bad, headers={"Idempotency-Key": "x"}).status_code == 422
    assert client.post("/api/v1/analyze", json=demo).status_code == 422


@pytest.mark.parametrize(
    "mutation", ["probability", "bbox", "mode", "dangling", "cycle", "span", "duplicate", "identity"]
)
def test_invariants(client, app, demo, mutation):
    result, _ = run(client, app, demo)
    a = result["analysis"]["atomic_claims"][0]
    v = result["analysis"]["visual_assessments"][0]
    if mutation == "probability":
        v["probabilities"] = {"Supported": 0.8, "Contradicted": 0.5, "Unobservable": 0}
    elif mutation == "bbox":
        v["supporting_regions"][0]["bbox"] = [0, 0, 2, 1]
    elif mutation == "mode":
        result["analysis"]["run"]["mode"] = "live"
    elif mutation == "dangling":
        a["depends_on"] = ["a999999"]
    elif mutation == "cycle":
        a["depends_on"] = [a["atom_id"]]
    elif mutation == "span":
        a["spans"][0]["end"] = 9999
    elif mutation == "duplicate":
        result["analysis"]["visual_assessments"].append(copy.deepcopy(v))
    elif mutation == "identity":
        result["input"]["image"]["sha256"] = "f" * 64
    with pytest.raises(ValidationError):
        AuroraBundle.model_validate(result)


def test_missing_is_not_null(demo):
    del demo["retrieval"]
    with pytest.raises(ValidationError):
        AuroraBundle.model_validate(demo)


def test_uri_not_identity(client, app, demo):
    result, _ = run(client, app, demo)
    result["input"]["image"]["uri"] = "media/portable.png"
    assert AuroraBundle.model_validate(result).analysis.atom_set_id == result["analysis"]["atom_set_id"]


def test_source_examples_are_actually_checked():
    # Existing handoff input without analysis is valid; preserves supplied material.
    value = strict_json(Path("contoh/01_INPUT_RETRIEVAL.json").read_bytes())
    assert AuroraBundle.model_validate(value).analysis is None
