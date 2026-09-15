"""Okapi BM25 over a local JSONL corpus — real local retrieval, no network."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from aurora_evidence.core.tokenize import tokenize


@dataclass
class CorpusDoc:
    doc_id: str
    title: str
    text: str
    publisher: str | None
    url: str | None
    language: str | None
    published_at: str | None
    kind: str  # fact_check | news | official | web | local_corpus | user_supplied
    independence_group: str | None = None
    synthetic: bool = False
    image_dhash: str | None = None
    image_sha256: str | None = None


@dataclass
class BM25Index:
    docs: list[CorpusDoc] = field(default_factory=list)
    _doc_ids: set[str] = field(default_factory=set)
    _tf: list[dict[str, int]] = field(default_factory=list)
    _df: dict[str, int] = field(default_factory=dict)
    _lengths: list[int] = field(default_factory=list)
    _avgdl: float = 0.0
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: list[CorpusDoc]) -> "BM25Index":
        index = cls(docs=docs)
        for doc in docs:
            if doc.doc_id in index._doc_ids:
                raise ValueError(f"Duplicate corpus doc id: {doc.doc_id}")
            index._doc_ids.add(doc.doc_id)
            counts: dict[str, int] = {}
            for token in tokenize(f"{doc.title} {doc.title} {doc.text}"):
                counts[token] = counts.get(token, 0) + 1
            index._tf.append(counts)
            index._lengths.append(sum(counts.values()))
            for token in counts:
                index._df[token] = index._df.get(token, 0) + 1
        total = sum(index._lengths)
        index._avgdl = total / len(docs) if docs else 0.0
        return index

    def search(self, query: str, limit: int = 10) -> list[tuple[CorpusDoc, float]]:
        """Return (doc, bm25_score) pairs, best first. Score is unbounded rank score."""
        tokens = tokenize(query)
        if not tokens or not self.docs:
            return []
        n = len(self.docs)
        scores: list[float] = []
        for i, counts in enumerate(self._tf):
            score = 0.0
            dl = self._lengths[i] or 1
            for token in tokens:
                if token not in counts:
                    continue
                df = self._df[token]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = counts[token]
                score += (
                    idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
                )
            scores.append(score)
        ranked = sorted(
            ((doc, score) for doc, score in zip(self.docs, scores) if score > 0),
            key=lambda pair: (-pair[1], pair[0].doc_id),
        )
        return ranked[:limit]


def load_corpus(path: Path) -> BM25Index:
    """Load a JSONL corpus; each line is one document (schema documented in docs/data.md)."""
    docs = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            docs.append(
                CorpusDoc(
                    doc_id=str(raw["id"]),
                    title=str(raw["title"]),
                    text=str(raw["text"]),
                    publisher=raw.get("publisher"),
                    url=raw.get("url"),
                    language=raw.get("language"),
                    published_at=raw.get("published_at"),
                    kind=raw.get("kind", "local_corpus"),
                    independence_group=raw.get("independence_group"),
                    synthetic=bool(raw.get("synthetic", False)),
                    image_dhash=raw.get("image_dhash"),
                    image_sha256=raw.get("image_sha256"),
                )
            )
    return BM25Index.build(docs)
