"""Real C2PA manifest validation round-trip: sign with a test CA, verify in screening.

The fixture generates a CA + leaf signing chain with openssl (as C2PA requires
keyUsage/EKU extensions), signs PNGs with c2pa-python intents, and checks that
``screen_image`` reports signature validation conservatively: only a validated
manifest with a trained-algorithmic digital source type can flag AI generation.
"""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import c2pa
import pytest
from aurora_visual.origin_screen import screen_image
from PIL import Image

requires_openssl = pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI required")


def _media(tag):
    return SimpleNamespace(asset_id="asset_" + tag * 64, sha256=tag * 64, media_type="image/png")


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    base = tmp_path_factory.mktemp("c2pa-chain")
    ca_key, ca_cert = base / "ca.key", base / "ca.pem"
    leaf_key, leaf_csr, leaf_cert = base / "leaf.key", base / "leaf.csr", base / "leaf.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-new",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:prime256v1",
            "-keyout",
            str(ca_key),
            "-out",
            str(ca_cert),
            "-nodes",
            "-subj",
            "/CN=Aurora Test CA/O=Aurora",
            "-days",
            "3650",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "openssl",
            "req",
            "-new",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:prime256v1",
            "-keyout",
            str(leaf_key),
            "-out",
            str(leaf_csr),
            "-nodes",
            "-subj",
            "/CN=Aurora Test Signer/O=Aurora",
        ],
        check=True,
        capture_output=True,
    )
    ext = base / "leaf.ext"
    ext.write_text(
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,nonRepudiation\n"
        "extendedKeyUsage=emailProtection\n"
        "subjectKeyIdentifier=hash\n"
        "authorityKeyIdentifier=keyid,issuer\n"
    )
    subprocess.run(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            str(leaf_csr),
            "-CA",
            str(ca_cert),
            "-CAkey",
            str(ca_key),
            "-out",
            str(leaf_cert),
            "-days",
            "3650",
            "-extfile",
            str(ext),
        ],
        check=True,
        capture_output=True,
    )
    return base, ca_cert, (leaf_cert.read_bytes() + ca_cert.read_bytes()), leaf_key


def _sign(base, chain_fixture, source: Path, target: Path, source_type) -> None:
    _, ca_cert, cert, leaf_key = chain_fixture
    signer_info = c2pa.C2paSignerInfo(
        alg=c2pa.C2paSigningAlg.ES256,
        sign_cert=cert,
        private_key=leaf_key.read_bytes(),
        ta_url=None,
    )
    ctx = c2pa.Context.from_dict(
        {
            "builder": {
                "claim_generator_info": {"name": "aurora-c2pa-fixture", "version": "1.0"},
            }
        }
    )
    with c2pa.Signer.from_info(signer_info) as signer:
        with c2pa.Builder({}, context=ctx) as builder:
            builder.set_intent(c2pa.C2paBuilderIntent.CREATE, source_type)
            builder.sign_file(str(source), str(target), signer)


@pytest.fixture(scope="module")
def signed_ai(tmp_path_factory, chain):
    base = tmp_path_factory.mktemp("c2pa-signed")
    plain = base / "plain.png"
    Image.new("RGB", (32, 32), "green").save(plain)
    target = base / "signed_ai.png"
    _sign(base, chain, plain, target, c2pa.C2paDigitalSourceType.TRAINED_ALGORITHMIC_MEDIA)
    return plain, target


@requires_openssl
def test_verified_manifest_with_trained_source_flags_ai_origin(signed_ai, chain):
    _, ca_cert, _, _ = chain
    _, target = signed_ai
    result = screen_image(target, _media("a"), "live", c2pa_trust_anchors=str(ca_cert))
    c2pa_report = result["provenance"]["c2pa"]
    assert c2pa_report["status"] == "verified"
    assert c2pa_report["verification"] == "signature_validated"
    assert c2pa_report["validation_status"] == []
    assert "trainedAlgorithmicMedia" in c2pa_report["digital_source_types"]
    assert c2pa_report["manifest_count"] == 1
    assert c2pa_report["claim_generator"].startswith("aurora-c2pa-fixture")
    assert c2pa_report["signer"]["issuer"] == "Aurora"
    assert result["decision"]["label"] == "likely_ai_generated"
    assert "C2PA" in result["decision"]["rationale"]


@requires_openssl
def test_camera_capture_manifest_does_not_flag_ai(signed_ai, chain):
    plain, _ = signed_ai
    capture = plain.with_name("signed_capture.png")
    _sign(plain.parent, chain, plain, capture, c2pa.C2paDigitalSourceType.DIGITAL_CAPTURE)
    _, ca_cert, _, _ = chain
    result = screen_image(capture, _media("e"), "live", c2pa_trust_anchors=str(ca_cert))
    assert result["provenance"]["c2pa"]["status"] == "verified"
    assert result["decision"]["label"] == "inconclusive"


@requires_openssl
def test_untrusted_signer_is_never_treated_as_validated(signed_ai):
    _, target = signed_ai
    # Default SDK trust store: our test CA is not a known C2PA trust anchor.
    result = screen_image(target, _media("b"), "live")
    c2pa_report = result["provenance"]["c2pa"]
    assert c2pa_report["status"] == "present_unverified"
    assert c2pa_report["verification"] == "not_verified"
    codes = [entry["code"] for entry in c2pa_report["validation_status"]]
    assert "signingCredential.untrusted" in codes
    assert result["decision"]["label"] == "inconclusive"


@requires_openssl
def test_tampered_manifest_fails_validation(signed_ai, chain):
    _, target = signed_ai
    tampered = target.with_name("tampered.png")
    data = bytearray(target.read_bytes())
    for index in range(len(data) // 2, len(data) // 2 + 64):
        data[index] ^= 0xFF
    tampered.write_bytes(bytes(data))
    _, ca_cert, _, _ = chain
    result = screen_image(tampered, _media("c"), "live", c2pa_trust_anchors=str(ca_cert))
    c2pa_report = result["provenance"]["c2pa"]
    assert c2pa_report["status"] == "present_unverified"
    codes = [entry["code"] for entry in c2pa_report["validation_status"]]
    assert any("hash" in code or "mismatch" in code for code in codes)
    assert result["decision"]["label"] == "inconclusive"


@requires_openssl
def test_plain_image_without_manifest_is_not_detected(signed_ai):
    plain, _ = signed_ai
    result = screen_image(plain, _media("d"), "live")
    assert result["provenance"]["c2pa"]["status"] == "not_detected"
    assert result["provenance"]["c2pa"]["manifest_count"] == 0


def test_misconfigured_trust_anchors_disable_validation(tmp_path):
    path = tmp_path / "plain.png"
    Image.new("RGB", (20, 20), "red").save(path)
    result = screen_image(path, _media("f"), "live", c2pa_trust_anchors=str(tmp_path / "missing.pem"))
    c2pa_report = result["provenance"]["c2pa"]
    assert c2pa_report["status"] == "unavailable"
    assert c2pa_report["validation_status"] == [{"code": "trust_anchors_misconfigured", "explanation": None}]
    assert result["decision"]["label"] == "inconclusive"


def test_manifest_report_is_json_serializable_and_bounded(signed_ai, chain):
    _, ca_cert, _, _ = chain
    _, target = signed_ai
    result = screen_image(target, _media("g"), "live", c2pa_trust_anchors=str(ca_cert))
    serialized = json.dumps(result)
    assert len(serialized) < 20_000
    assert "BEGIN CERTIFICATE" not in serialized
