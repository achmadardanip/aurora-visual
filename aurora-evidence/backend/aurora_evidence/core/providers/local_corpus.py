"""Local corpus fact-check provider: real BM25 search over the loaded JSONL corpus."""

import time

from aurora_evidence.core.bm25 import BM25Index
from aurora_evidence.core.providers.interfaces import FactCheckProvider, SearchHit
from aurora_evidence.core.tokenize import sentences


class LocalCorpusProvider(FactCheckProvider):
    name = "local-corpus"
    capability = "fact_check_search"

    def __init__(self, index: BM25Index):
        self.index = index

    def status(self, settings) -> str:
        return "ok" if self.index.docs else "unconfigured"

    def search(self, claim: str, language: str, as_of, settings, budget_ms: int) -> list[SearchHit]:
        started = time.perf_counter()
        hits = []
        for doc, score in self.index.search(claim, limit=settings.corpus_hits):
            if time.perf_counter() - started > budget_ms / 1000:
                break
            # Temporal eligibility under a strict as_of cutoff.
            if as_of and doc.published_at and doc.published_at > as_of:
                continue
            text = doc.text
            # Honest excerpt: the first sentence overlapping query terms, else the opening.
            excerpt = ""
            for sentence in sentences(text):
                if any(word.lower() in sentence.lower() for word in claim.split()[:4]):
                    excerpt = sentence
                    break
            hits.append(
                SearchHit(
                    title=doc.title,
                    url=doc.url,
                    snippet=excerpt or (sentences(text)[0] if sentences(text) else ""),
                    publisher=doc.publisher,
                    language=doc.language,
                    published_at=doc.published_at,
                    kind=doc.kind,
                    rank=len(hits) + 1,
                    independence_group=doc.independence_group,
                    full_text=text,
                    doc_id=doc.doc_id,
                )
            )
        return hits
