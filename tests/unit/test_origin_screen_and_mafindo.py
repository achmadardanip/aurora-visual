import io
import json
from types import SimpleNamespace

import httpx
from aurora_visual.mafindo import MafindoClient, redact_url
from aurora_visual.origin_screen import screen_image
from PIL import Image, PngImagePlugin


def test_origin_screen_marks_generator_metadata_without_deciding_truth(tmp_path):
    path = tmp_path / "screened.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Software", "Stable Diffusion test export")
    Image.new("RGB", (20, 20), "red").save(path, pnginfo=metadata)
    result = screen_image(
        path,
        SimpleNamespace(asset_id="asset_" + "a" * 64, sha256="a" * 64),
        "live",
    )
    assert result["decision"]["label"] == "likely_ai_generated"
    assert result["decision"]["does_not_affect_visual_assessment"]
    assert result["decision"]["does_not_decide_claim_truth"]
    assert result["detectors"][0]["status"] == "unavailable"


def test_origin_screen_reports_c2pa_marker_as_unverified(tmp_path):
    path = tmp_path / "marker.png"
    raw = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(raw, "PNG")
    path.write_bytes(raw.getvalue() + b"c2pa")
    result = screen_image(
        path,
        SimpleNamespace(asset_id="asset_" + "b" * 64, sha256="b" * 64),
        "live",
    )
    assert result["provenance"]["c2pa"]["status"] == "marker_present"
    assert result["provenance"]["c2pa"]["verification"] == "not_verified"


def test_origin_screen_camera_metadata_is_not_a_negative_detector_result(tmp_path):
    path = tmp_path / "camera.jpg"
    exif = Image.Exif()
    exif[271] = "Camera Maker"
    exif[272] = "Camera Model"
    Image.new("RGB", (20, 20), "red").save(path, exif=exif)
    result = screen_image(
        path,
        SimpleNamespace(asset_id="asset_" + "c" * 64, sha256="c" * 64),
        "live",
    )
    assert result["decision"]["label"] == "no_strong_ai_signal"
    assert result["metadata"]["camera_metadata_present"]
    assert all(detector["status"] == "unavailable" for detector in result["detectors"])
    assert "bukan bukti" in result["limitations"][2].lower()


def test_origin_screen_invalid_file_is_bounded(tmp_path):
    path = tmp_path / "invalid.bin"
    path.write_bytes(b"not an image")
    result = screen_image(
        path,
        SimpleNamespace(asset_id="asset_" + "d" * 64, sha256="d" * 64),
        "live",
    )
    assert result["metadata"]["status"] == "unavailable"
    assert result["decision"]["label"] == "inconclusive"
    assert result["provenance"]["watermark"]["status"] == "unavailable"


def test_mafindo_client_normalizes_response_and_hides_credential():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(
            200,
            json=[
                {
                    "id": 17,
                    "title": "Contoh provider",
                    "status": 2,
                    "classification": "Disinformasi",
                    "tanggal": "2026-09-13",
                }
            ],
        )

    client = MafindoClient("test-key", transport=httpx.MockTransport(handler))
    result = client.search("title", "teks dengan / karakter")
    assert seen and "%2F" in seen[0]
    assert result["status"] == "ok"
    assert result["results"] == [
        {
            "id": "17",
            "title": "Contoh provider",
            "classification": "Disinformasi",
            "provider_status": 2,
            "published_at": "2026-09-13",
        }
    ]
    assert "test-key" not in json.dumps(result)


def test_mafindo_failure_and_redaction_never_surface_key():
    client = MafindoClient(
        "test-key",
        transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("down"))),
    )
    result = client.latest()
    assert result["status"] == "failed"
    assert "test-key" not in json.dumps(result)
    assert redact_url("https://example.test/a/test-key", "test-key").endswith("/[REDACTED]")


def test_mafindo_requires_server_configuration_without_requesting_provider():
    client = MafindoClient("")
    result = client.latest()
    assert result["status"] == "unconfigured"
    assert result["results"] == []


def test_mafindo_rejects_invalid_diagnostic_inputs():
    client = MafindoClient("test-key")
    for limit in (0, 21, "1"):
        try:
            client.latest(limit)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid limit was accepted")
    for field, value in (("unknown", "value"), ("title", ""), ("title", "x" * 501)):
        try:
            client.search(field, value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid search was accepted")


def test_mafindo_limits_provider_response_and_hides_key_on_http_failure():
    def handler(request):
        assert "test-key" in str(request.url)
        return httpx.Response(503, text="provider unavailable")

    client = MafindoClient("test-key", transport=httpx.MockTransport(handler))
    result = client.latest()
    assert result["status"] == "failed"
    assert "test-key" not in json.dumps(result)


def test_mafindo_limits_rows_redacts_encoded_key_and_rejects_nested_status():
    api_key = "test key/with?"
    encoded_key = "test%20key%2Fwith%3F"

    def handler(request):
        return httpx.Response(
            200,
            json=[
                {
                    "id": index,
                    "title": encoded_key + " " + "x" * 700,
                    "classification": api_key,
                    "status": {"credential": api_key},
                }
                for index in range(25)
            ],
        )

    result = MafindoClient(api_key, transport=httpx.MockTransport(handler)).latest(20)
    serialized = json.dumps(result)
    assert result["status"] == "ok"
    assert len(result["results"]) == 20
    assert len(result["results"][0]["title"]) == 500
    assert result["results"][0]["provider_status"] is None
    assert api_key not in serialized
    assert encoded_key not in serialized


def test_mafindo_rejects_oversized_provider_response_without_leaking_key():
    client = MafindoClient(
        "test-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 2_000_001)),
    )
    result = client.latest()
    assert result["status"] == "failed"
    assert "test-key" not in json.dumps(result)
