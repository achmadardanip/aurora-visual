"""Core module-2 tests: BM25, dedup, ranking, query planning, temporal rules."""

from pathlib import Path

import pytest
from aurora_evidence.core.bm25 import BM25Index, CorpusDoc
from aurora_evidence.core.dedup import canonical_url, cluster_evidence, dhash, hamming_hex
from aurora_evidence.core.queries import plan_queries
from aurora_evidence.core.ranking import reciprocal_rank_fusion, rerank
from aurora_evidence.core.retrieval import _temporal_eligible
from aurora_evidence.core.tokenize import numbers, tokenize
from PIL import Image


def doc(doc_id, title, text, **kwargs):
    return CorpusDoc(
        doc_id=doc_id,
        title=title,
        text=text,
        publisher="p",
        url=None,
        language="id",
        published_at=None,
        kind="news",
        **kwargs,
    )


def test_bm25_ranks_relevant_document_first():
    index = BM25Index.build(
        [
            doc("d1", "energi terbarukan", "kebijakan energi terbarukan menargetkan pembangkit listrik"),
            doc("d2", "olahraga", "pertandingan final bola berakhir dengan skor tiga banding satu"),
        ]
    )
    results = index.search("kebijakan energi terbarukan")
    assert results and results[0][0].doc_id == "d1"
    assert index.search("pertandingan sepak bola")[0][0].doc_id == "d2"


def test_bm25_empty_or_no_match_returns_empty():
    index = BM25Index.build([doc("d1", "x", "isi dokumen")])
    assert index.search("") == []
    assert index.search("zzzznotpresent") == []


def test_duplicate_doc_id_rejected():
    with pytest.raises(ValueError):
        BM25Index.build([doc("d1", "a", "x"), doc("d1", "b", "y")])


def test_canonical_url_strips_tracking_and_normalizes():
    assert canonical_url("https://Example.com/path/?utm_source=x&id=1") == "https://example.com/path?id=1"
    assert canonical_url(None) is None


def test_cluster_same_url_and_text_share_group():
    items = [
        {
            "source": {"url": "https://a.example/x?utm_source=t"},
            "content": {"text": "berita sama"},
            "provenance": {},
        },
        {"source": {"url": "https://a.example/x"}, "content": {"text": "berita sama"}, "provenance": {}},
        {
            "source": {"url": "https://b.example/y"},
            "content": {"text": "laporan berbeda total tentang hal lain"},
            "provenance": {},
        },
    ]
    cluster_evidence(items)
    assert items[0]["provenance"]["duplicate_cluster_id"] == items[1]["provenance"]["duplicate_cluster_id"]
    assert items[0]["provenance"]["independence_group_id"] != items[2]["provenance"]["independence_group_id"]


def test_near_duplicate_text_clusters_via_shingles():
    text = " ".join(f"kata{i}" for i in range(20))
    near = text.replace("kata19", "kata19")
    items = [
        {"source": {"url": "https://a.example/1"}, "content": {"text": text}, "provenance": {}},
        {"source": {"url": "https://b.example/2"}, "content": {"text": near}, "provenance": {}},
    ]
    cluster_evidence(items)
    assert items[0]["provenance"]["duplicate_cluster_id"] == items[1]["provenance"]["duplicate_cluster_id"]


def test_declared_independence_group_preserved():
    items = [
        {
            "source": {"url": "https://a.example/1"},
            "content": {"text": "berita syndication"},
            "provenance": {"independence_group_override": "igr-manual"},
        },
        {
            "source": {"url": "https://b.example/2"},
            "content": {"text": "berita syndication"},
            "provenance": {},
        },
    ]
    cluster_evidence(items)
    assert items[0]["provenance"]["independence_group_id"] == "igr-manual"
    assert items[1]["provenance"]["independence_group_id"] != "igr-manual"


def test_dhash_identical_images_zero_hamming():
    image = Image.new("RGB", (64, 48), "#e02222")
    assert hamming_hex(dhash(image), dhash(image.copy())) == 0


def test_rrf_prefers_consistently_ranked_items():
    scores = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    assert scores["a"] > scores["b"] and scores["a"] > scores["c"]


def test_rerank_fills_scores_and_respects_group_cap():
    items = {}
    for i in range(6):
        evidence_id = f"ev_{'0' * 32}_{i:06d}"
        items[evidence_id] = {
            "evidence_id": evidence_id,
            "atom_ids": ["a000001"],
            "source": {"kind": "fact_check"},
            "content": {"status": "full"},
            "provenance": {"independence_group_id": "g1" if i < 4 else "g2", "temporal_eligible": True},
            "relevance_score": None,
            "credibility_score": None,
            "rerank_score": None,
            "score_breakdown": {},
        }
    selected = rerank(items, {k: 1.0 for k in items}, limit=10, group_cap=2)
    assert len(selected) == 4  # 2 from g1 + 2 from g2
    assert all(item["relevance_score"] is not None and 0 <= item["relevance_score"] <= 1 for item in selected)
    assert all("rrf" in item["score_breakdown"] for item in selected)


def test_query_planning_without_atoms_uses_caption():
    queries = plan_queries("Presiden menetapkan 30 September sebagai libur nasional", None, None)
    assert queries[0].variant == "caption"
    assert queries[1].variant == "fact_check"
    assert len(queries) <= 8
    assert all(q.atom_ids == [] for q in queries)


def test_temporal_eligibility_rules():
    assert _temporal_eligible("2026-01-01T00:00:00+00:00", None) is True
    assert _temporal_eligible(None, "2026-01-01T00:00:00+00:00") is None
    assert _temporal_eligible("2025-12-31T23:59:59+00:00", "2026-01-01T00:00:00+00:00") is True
    assert _temporal_eligible("2026-01-02T00:00:00+00:00", "2026-01-01T00:00:00+00:00") is False


def test_numbers_extracts_digits_and_words():
    assert "30" in numbers("tanggal 30 September")
    assert "2" in numbers("dua orang hadir")


def test_tokenize_lowercase_and_stopwords():
    tokens = tokenize("Gedung Sate dibangun pada tahun 1920")
    assert "dibangun" in tokens and "pada" not in tokens
    assert "1920" in tokens


def test_load_corpus_from_fixture():
    from aurora_evidence.core.bm25 import load_corpus

    index = load_corpus(
        Path(__file__).resolve().parents[2] / "aurora-evidence" / "fixtures" / "demo_corpus.jsonl"
    )
    assert len(index.docs) >= 6
    results = index.search("hari libur nasional 30 September")
    assert results and "libur" in results[0][0].title.lower()
