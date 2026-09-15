"""Deterministic query planning from caption, atoms and OCR.

Baseline (mandatory, no LLM): caption query, fact-check style variant, per-atom
statements, and entity/number queries. Every transformation is recorded so the
query log shows exactly what was sent to each provider.
"""

import re
from dataclasses import dataclass

from aurora_evidence.core.tokenize import tokenize

CAPITALIZED = re.compile(r"\b[A-Z][a-zA-Zà-ÿ]{2,}\b")
DIGITS = re.compile(r"\b\d+(?:[.,]\d+)*\b")

MAX_QUERIES = 8


@dataclass
class PlannedQuery:
    query_id: str
    query: str
    variant: str  # caption | fact_check | atom | entity | temporal
    atom_ids: list[str]


def plan_queries(claim_text: str, atoms=None, ocr_texts=None) -> list[PlannedQuery]:
    """Build a bounded, deterministic query set. atoms/ocr may be None or empty."""
    queries: list[PlannedQuery] = [
        PlannedQuery("q000001", claim_text.strip(), "caption", []),
        PlannedQuery("q000002", f'fakta cek "{claim_text.strip()}"', "fact_check", []),
    ]
    for atom in atoms or []:
        if len(queries) >= MAX_QUERIES:
            break
        statement = atom.statement.strip()
        if statement and statement != claim_text.strip():
            queries.append(PlannedQuery("", statement, "atom", [atom.atom_id]))
    entities = sorted({m.group(0) for m in CAPITALIZED.finditer(claim_text)})
    numbers = sorted({m.group(0) for m in DIGITS.finditer(claim_text)})
    if (entities or numbers) and len(queries) < MAX_QUERIES:
        queries.append(PlannedQuery("", " ".join(entities[:4] + numbers[:3]), "entity", []))
    for text in (ocr_texts or [])[:1]:
        if len(queries) < MAX_QUERIES and text.strip():
            queries.append(PlannedQuery("", text.strip()[:200], "temporal", []))
    for index, query in enumerate(queries, start=1):
        query.query_id = f"q{index:06d}"
    return queries[:MAX_QUERIES]


def query_tokens(query: str) -> list[str]:
    return tokenize(query)
