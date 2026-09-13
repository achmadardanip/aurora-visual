"""MAFINDO V2 corpus puller, LLM judge, and Optuna tuning: mocked + smoke tests."""

import json
from pathlib import Path

import httpx
import pytest
from aurora_visual.mafindo_corpus import (
    CorpusError,
    MafindoCorpusClient,
    normalize_record,
    pull_corpus,
    strip_html,
)
from aurora_visual.training.data import smoke_samples
from aurora_visual.training.llm_judge import (
    OllamaJudge,
    judge_weak_pairs,
    triage_corpus,
)
from aurora_visual.training.tuning import SEARCH_SPACE, suggest_config, tune


def _pages(records, page_size=2):
    def handler(request):
        body = dict(pair.split("=") for pair in request.read().decode().split("&"))
        offset, limit = int(body["offset"]), int(body["limit"])
        return httpx.Response(200, json=records[offset : offset + limit])

    return handler


def _records(count=5):
    return [
        {
            "id": str(100 - i),
            "authors": "1",
            "status": "2",
            "classification": "Fabricated Content",
            "title": f"[SALAH] Judul uji {i}",
            "content": "<p>Isi <b>uji</b> nomor %d</p>" % i,
            "fact": "<p>Fakta uji</p>",
            "conclusion": "Simpulan",
            "references": "https://example.test/r",
            "source_issue": "Facebook",
            "source_link": "https://facebook.example/%d" % i,
            "picture1": "https://cdn.example/%d.jpg" % i,
            "picture2": "",
            "tanggal": "2026-09-01",
            "tags": "cekfakta",
            "category": "Politik",
        }
        for i in range(count)
    ]


def test_mafindo_v2_client_pagination_and_total():
    records = _records(5)
    client = MafindoCorpusClient(
        "key",
        transport=httpx.MockTransport(
            lambda request: (
                httpx.Response(200, text="5") if "get_total" in str(request.url) else _pages(records)(request)
            )
        ),
    )
    assert client.get_total() == 5
    page = client.list_page(2, 0)
    assert [row["id"] for row in page] == ["100", "99"]
    assert [row["id"] for row in client.list_page(2, 2)] == ["98", "97"]


def test_pull_corpus_deduplicates_and_resumes(tmp_path):
    records = _records(5)
    transport = httpx.MockTransport(
        lambda request: (
            httpx.Response(200, text="5") if "get_total" in str(request.url) else _pages(records)(request)
        )
    )
    output = tmp_path / "corpus.jsonl"
    summary = pull_corpus("key", output, page_size=2, delay=0, transport=transport)
    assert summary["unique_records"] == 5
    assert summary["duplicates_skipped"] == 0
    assert summary["finished"] is True
    assert "bukan" not in summary["usage"]  # usage note stays factual
    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(lines) == 5 and len({line["id"] for line in lines}) == 5
    assert lines[0]["content_plain"] == "Isi uji nomor 0"
    assert lines[0]["pulled_at"].startswith("20")
    # A second completed run is a no-op: the saved state short-circuits before
    # any new provider request (request counter unchanged from the first run).
    again = pull_corpus("key", output, page_size=2, delay=0, transport=transport)
    assert again["unique_records"] == 5
    assert again["requests"] == summary["requests"]
    assert again["finished"] is True


def test_pull_corpus_respects_max_records(tmp_path):
    records = _records(6)
    transport = httpx.MockTransport(
        lambda request: (
            httpx.Response(200, text="6") if "get_total" in str(request.url) else _pages(records)(request)
        )
    )
    output = tmp_path / "corpus.jsonl"
    summary = pull_corpus("key", output, max_records=3, page_size=2, delay=0, transport=transport)
    assert summary["unique_records"] == 3


def test_pull_corpus_never_leaks_the_api_key(tmp_path):
    records = _records(2)
    transport = httpx.MockTransport(
        lambda request: (
            httpx.Response(200, text="2") if "get_total" in str(request.url) else _pages(records)(request)
        )
    )
    output = tmp_path / "corpus.jsonl"
    pull_corpus("secret-key/42", output, page_size=2, delay=0, transport=transport)
    assert "secret-key" not in output.read_text()


def test_pull_corpus_unconfigured_raises():
    with pytest.raises(CorpusError, match="unconfigured"):
        pull_corpus("", Path("/tmp/never.jsonl"))


def test_normalize_record_bounds_and_redacts():
    row = _records(1)[0]
    row["title"] = "secret-key judul panjang " + "x" * 20_000
    record = normalize_record(row, "secret-key")
    assert record is not None and len(record["title"]) <= 10_000
    assert "secret-key" not in record["title"]
    assert normalize_record({"id": ""}, "key") is None


def test_strip_html_bounded():
    assert strip_html("<p>halo <b>dunia</b></p>") == "halo dunia"
    assert len(strip_html("<p>" + "a" * 50_000 + "</p>")) == 10_000
    assert strip_html(None) == ""


def _judge_handler(responses):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"message": {"content": json.dumps(responses.pop(0))}})

    return handler, calls


def test_judge_weak_pairs_accepts_consistent_verdicts():
    verdicts = [
        {"meaning_preserved": True, "claim_changed": False, "explanation": "paraphrase", "confidence": 0.9},
        {
            "meaning_preserved": False,
            "claim_changed": True,
            "explanation": "count differs",
            "confidence": 0.8,
        },
    ]
    handler, calls = _judge_handler(list(verdicts))
    judge = OllamaJudge(
        url="http://127.0.0.1:11434",
        model="qwen-test",
        allowed_origins=["http://127.0.0.1:11434"],
        transport=httpx.MockTransport(handler),
    )
    pairs = [
        {"source": "Kucing hitam", "modified": "Kucing hitam.", "kind": "paraphrase"},
        {"source": "dua anjing", "modified": "tiga anjing", "kind": "number_change"},
    ]
    judged = judge_weak_pairs(judge, pairs)
    assert judged[0]["judge"]["disposition"] == "accepted"
    assert judged[1]["judge"]["disposition"] == "accepted"
    assert judged[0]["judge"]["provenance"] == "llm-judge-v1"
    assert calls[0].read().count(b"qwen-test") == 1


def test_judge_weak_pairs_rejects_inconsistent_verdicts():
    # A paraphrase the judge says changed meaning must be rejected, not relabeled.
    verdicts = [
        {"meaning_preserved": False, "claim_changed": True, "explanation": "?", "confidence": 0.5},
    ]
    handler, _ = _judge_handler(list(verdicts))
    judge = OllamaJudge(
        url="http://127.0.0.1:11434",
        model="qwen-test",
        allowed_origins=["http://127.0.0.1:11434"],
        transport=httpx.MockTransport(handler),
    )
    judged = judge_weak_pairs(judge, [{"source": "a", "modified": "a.", "kind": "paraphrase"}])
    assert judged[0]["judge"]["disposition"] == "rejected"


def test_judge_rejects_malformed_verdicts():
    handler, _ = _judge_handler([{"meaning_preserved": "yes"}])
    judge = OllamaJudge(
        url="http://127.0.0.1:11434",
        model="qwen-test",
        allowed_origins=["http://127.0.0.1:11434"],
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError):
        judge.judge_pair("a", "b", "synonym")


def test_judge_enforces_origin_allowlist():
    judge = OllamaJudge(
        url="http://evil.example:11434", model="m", allowed_origins=["http://127.0.0.1:11434"]
    )
    with pytest.raises(ValueError, match="allowlisted"):
        judge.judge_pair("a", "b", "synonym")


def test_triage_corpus_records_and_skips_failures():
    verdicts = [
        {"is_visual_claim": True, "parseable": True, "claim_kinds": ["action", "location"], "note": "ok"},
    ]
    handler, _ = _judge_handler(list(verdicts))
    judge = OllamaJudge(
        url="http://127.0.0.1:11434",
        model="qwen-test",
        allowed_origins=["http://127.0.0.1:11434"],
        transport=httpx.MockTransport(handler),
    )
    records = [{"id": "1", "title": "Massa berunjuk rasa di Monas"}, {"id": "2", "title": ""}]
    result = triage_corpus(judge, records, 10)
    assert len(result) == 1
    assert result[0]["judge"]["claim_kinds"] == ["action", "location"]


def test_tune_runs_smoke_trials_on_fixture(tmp_path):
    samples = smoke_samples(17)
    study, best_config, summary = tune(samples, trials=3, base_config={"epochs": 2, "seed": 17})
    assert summary["trials"] == 3
    assert 0.0 <= summary["best_value_validation_macro_f1"] <= 1.0
    assert set(best_config) == {
        "learning_rate",
        "hidden",
        "method",
        "use_unmatched",
        "cosine_only",
        "loss_weights",
    }
    assert set(best_config["loss_weights"]) == set(SEARCH_SPACE["loss_weights"])
    assert summary["data_kind"] == "fixture"
    assert "not a research result" in summary["note"]
    assert all(t["state"] == "COMPLETE" for t in summary["trials_summary"])


def test_tune_respects_minimum_trials():
    with pytest.raises(ValueError, match="positive"):
        tune(smoke_samples(17), trials=0, base_config={"epochs": 1})


def test_suggest_config_covers_search_space():
    class FakeTrial:
        def __init__(self):
            self.params = {}

        def suggest_float(self, name, low, high, log=False):
            self.params[name] = (low + high) / 2
            return self.params[name]

        def suggest_categorical(self, name, values):
            self.params[name] = values[0]
            return values[0]

    trial = FakeTrial()
    config = suggest_config(trial)
    assert config["hidden"] == SEARCH_SPACE["hidden"][0]
    assert set(config["loss_weights"]) == set(SEARCH_SPACE["loss_weights"])
