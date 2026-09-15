"""Demo provider: deterministic, clearly-labeled fixtures for demo mode.

Never used in live mode. Fixture documents carry SYNTHETIC labels so they can
never be mistaken for real evidence in reports or evaluation.
"""

from aurora_evidence.core.providers.interfaces import FactCheckProvider, SearchHit

DEMO_DOCS = [
    {
        "id": "demo-fc-1",
        "title": "[DEMO SINTETIS] Fakta cek: bidang merah pada gambar uji",
        "text": (
            "Periksa Fakta Demo memeriksa unggahan yang menyebut bidang berwarna merah. "
            "Pemeriksaan pada gambar sintetis demo menunjukkan bidang berwarna merah. "
            "Artikel ini adalah fixture demo dan bukan hasil pemeriksaan faktual nyata."
        ),
        "publisher": "Periksa Fakta Demo",
        "kind": "fact_check",
        "language": "id",
        "published_at": "2026-09-01T08:00:00+00:00",
    },
    {
        "id": "demo-news-1",
        "title": "[DEMO SINTETIS] Berita demo tentang bidang biru",
        "text": (
            "Kantor berita demo melaporkan bidang berwarna biru sesuai laporan photographer demo. "
            "Laporan ini dibuat untuk pengujian alur aplikasi dan bukan berita nyata."
        ),
        "publisher": "Kantor Berita Demo",
        "kind": "news",
        "language": "id",
        "published_at": "2026-09-02T09:30:00+00:00",
    },
]


class DemoFactCheckProvider(FactCheckProvider):
    name = "demo-fixtures"
    capability = "fact_check_search"

    def status(self, settings) -> str:
        return "ok"

    def search(self, claim: str, language: str, as_of, settings, budget_ms: int) -> list[SearchHit]:
        hits = []
        for rank, doc in enumerate(DEMO_DOCS, start=1):
            if as_of and doc["published_at"] > as_of:
                continue
            tokens = [w for w in claim.lower().split() if len(w) > 3][:4]
            snippet = doc["text"].split(". ")[1] if len(doc["text"].split(". ")) > 1 else doc["text"]
            if any(word in doc["text"].lower() for word in tokens):
                hits.append(
                    SearchHit(
                        title=doc["title"],
                        url=None,
                        snippet=snippet,
                        publisher=doc["publisher"],
                        language=doc["language"],
                        published_at=doc["published_at"],
                        kind=doc["kind"],
                        rank=len(hits) + 1,
                    )
                )
        return hits


def demo_forensic_fixtures(claim_text: str, sha) -> list[dict]:
    """Deterministic forensic signals for demo mode — labeled, never live."""
    from aurora_evidence.contract import sha as _sha

    return [
        {
            "modality": "text",
            "provider": "demo-fixtures",
            "target": {"kind": "claim_text", "asset_id": None, "text_sha256": _sha(claim_text)},
            "status": "ok",
            "raw_score": 0.12,
            "raw_scale": {"min": 0.0, "max": 1.0, "higher_means_ai": True},
            "raw_label": "human",
            "model_version": "demo-fixture-v1",
            "limitations": ["Fixture demo; bukan hasil detektor nyata"],
        }
    ]
