"""Shared text utilities for retrieval: tokenization, stopwords, shingles."""

import re

TOKEN = re.compile(r"[0-9a-zà-ÿ_]+", re.IGNORECASE)

# Small auditable stopword list (Indonesian + English) for BM25/overlap only.
# Not a linguistic model; documented limitation.
STOPWORDS = {
    "dan",
    "yang",
    "di",
    "ke",
    "dari",
    "untuk",
    "pada",
    "dengan",
    "ini",
    "itu",
    "adalah",
    "ada",
    "tidak",
    "bukan",
    "akan",
    "telah",
    "sudah",
    "dalam",
    "oleh",
    "juga",
    "atau",
    "karena",
    "sebagai",
    "bahwa",
    "para",
    "per",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "and",
    "or",
    "that",
    "this",
    "it",
    "as",
    "by",
}

NUMBER_WORDS = {
    "nol": 0,
    "satu": 1,
    "dua": 2,
    "tiga": 3,
    "empat": 4,
    "lima": 5,
    "enam": 6,
    "tujuh": 7,
    "delapan": 8,
    "sembilan": 9,
    "sepuluh": 10,
    "sebelas": 11,
    "seratus": 100,
    "seribu": 1000,
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "hundred": 100,
    "thousand": 1000,
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; digits and diacritics preserved, stopwords kept out."""
    return [t for t in (m.group(0).lower() for m in TOKEN.finditer(text or "")) if t not in STOPWORDS]


def content_tokens(text: str) -> list[str]:
    """Tokens including stopwords, for quote/sentence matching."""
    return [m.group(0).lower() for m in TOKEN.finditer(text or "")]


def numbers(text: str) -> list[str]:
    """Numeric mentions: digits (with separators) plus Indonesian/English number words."""
    found = [
        m.group(0).replace(".", "").replace(",", "") for m in re.finditer(r"\d+(?:[.,]\d+)*", text or "")
    ]
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", (text or "").lower()):
            found.append(str(value))
    return found


def years(text: str) -> list[str]:
    return [m.group(0) for m in re.finditer(r"\b(19|20)\d{2}\b", text or "")]


def sentences(text: str) -> list[str]:
    """Naive sentence split; quotes must remain exact substrings of the input."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = tokenize(text)
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0
