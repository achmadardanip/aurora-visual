"""Local image provenance index: dHash matching against corpus images.

This is local corpus search, NOT web-wide reverse image search. Corpus
documents may declare image_sha256/image_dhash; an uploaded asset is matched
exact (same sha256) or near-duplicate (dHash hamming <= 10).
"""

from aurora_evidence.core.bm25 import BM25Index
from aurora_evidence.core.dedup import dhash, hamming_hex, match_type_for
from aurora_evidence.core.providers.interfaces import ImageMatchHit, ImageProvenanceProvider


class LocalImageIndexProvider(ImageProvenanceProvider):
    name = "local-image-index"
    capability = "local_image_index"

    def __init__(self, index: BM25Index):
        self.index = index

    def status(self, settings) -> str:
        return "ok" if any(doc.image_dhash for doc in self.index.docs) else "unconfigured"

    def search(self, image, settings) -> list[ImageMatchHit]:
        """image: PIL Image of the uploaded asset (original bytes decoded)."""
        if not image:
            return []
        try:
            query_hash = dhash(image)
        except Exception:
            return []
        matches = []
        for doc in self.index.docs:
            if not doc.image_dhash:
                continue
            hamming = hamming_hex(query_hash, doc.image_dhash)
            match_type = match_type_for(hamming)
            if match_type != "semantic":
                matches.append(
                    ImageMatchHit(
                        matched_url=doc.url,
                        match_type=match_type,
                        score=round(1.0 - hamming / 64.0, 6),
                        title=doc.title,
                        publisher=doc.publisher,
                        published_at=doc.published_at,
                    )
                )
        matches.sort(key=lambda m: (-m.score, m.matched_url or ""))
        return matches[: settings.image_matches]
