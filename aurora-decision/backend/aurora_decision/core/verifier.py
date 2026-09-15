"""Deterministic EvidenceVerifier baseline: narrow explicit relations only.

Honest scope (documented limitation): entity/event token overlap, windowed
negation markers around the predicate, number/year conflicts on the same
topic. Anything the rules cannot decide returns Unclear — never a guess.
Stance per document is kept separate from the atom's factual label. Quotes
are exact substrings of the evidence content or the link is rejected.
"""

from dataclasses import dataclass

from aurora_decision.contract import Evidence
from aurora_decision.core.tokenize import content_tokens, numbers, sentences, tokenize

# Negation markers that can invert a predicate when adjacent to it.
NEGATION_WORDS = {"tidak", "bukan", "bukanlah", "tanpa", "no", "not", "never"}
# Attribution markers: a sentence reporting a claim does not verify the claim.
ATTRIBUTION_WORDS = ("mengklaim", "diklaim", "mengaku", "menurut unggahan", "beredar", "kabarnya", "isu")


@dataclass
class StanceResult:
    stance: str  # Supports | Contradicts | NotRelevant | Unclear
    quote: str | None
    rationale: str
    rule_version: str = "deterministic-v2"


def _atom_tokens(atom) -> set[str]:
    tokens = set(tokenize(atom.statement))
    for part in (atom.subject, atom.object):
        if part:
            tokens |= set(tokenize(part))
    return tokens


def _predicate_words(atom) -> list[str]:
    words = [w for w in tokenize(atom.predicate) if len(w) > 3]
    if not words:
        words = [w for w in tokenize(atom.statement) if len(w) > 3][:2]
    return words


def _windowed_negation(sentence_lower: str, predicate: str) -> bool:
    """Negation counts only within 3 words before the predicate occurrence."""
    words = sentence_lower.split()
    for index, word in enumerate(words):
        if predicate in word:
            window = words[max(0, index - 3) : index]
            return any(w.strip(",.") in NEGATION_WORDS for w in window)
    return False


def verify_atom_against_evidence(atom, evidence: Evidence) -> StanceResult:
    """Assess one atom against one eligible evidence document."""
    text = evidence.content.text
    if not text.strip() or evidence.content.status == "unavailable":
        return StanceResult("NotRelevant", None, "Konten bukti tidak tersedia.")
    atom_tokens = _atom_tokens(atom)
    if not atom_tokens:
        return StanceResult("Unclear", None, "Atom tanpa token yang dapat dicocokkan.")

    # Strongest overlap sentence becomes the candidate quote.
    best_sentence, best_overlap = "", 0
    for sentence in sentences(text):
        overlap = len(atom_tokens & set(tokenize(sentence)))
        if overlap > best_overlap:
            best_sentence, best_overlap = sentence, overlap
    if best_overlap < max(2, len(atom_tokens) // 3):
        return StanceResult(
            "NotRelevant",
            None,
            "Tidak ada tumpang tindih topik yang cukup antara atom dan dokumen.",
        )
    same_topic = best_overlap >= max(3, (len(atom_tokens) * 2) // 3)
    sentence_lower = best_sentence.lower()

    # Number/year conflict: the same quantity dimension with a different value.
    atom_numbers = {n for n in numbers(atom.statement) if len(n) >= 1}
    sentence_numbers = {n for n in numbers(best_sentence) if len(n) >= 1}
    predicates = _predicate_words(atom)
    predicate_present = any(p in content_tokens(best_sentence) for p in predicates)

    if same_topic and atom_numbers and sentence_numbers and not (atom_numbers & sentence_numbers):
        return StanceResult(
            "Contradicts",
            best_sentence,
            "Sumber membahas kuantitas/tahun pada topik yang sama dengan angka yang berbeda dari atom.",
        )

    if same_topic and predicate_present:
        negated_here = any(_windowed_negation(sentence_lower, p) for p in predicates)
        if negated_here and not atom.qualifiers.negated:
            return StanceResult(
                "Contradicts",
                best_sentence,
                "Kalimat sumber menegasikan predikat atom secara langsung pada topik yang sama.",
            )
        if negated_here and atom.qualifiers.negated:
            return StanceResult(
                "Supports",
                best_sentence,
                "Atom dinegasikan dan sumber juga menegasikan predikat yang sama.",
            )
        if not negated_here and atom.qualifiers.negated:
            return StanceResult(
                "Contradicts",
                best_sentence,
                "Atom dinegasikan sementara sumber menyatakan predikat tersebut positif.",
            )
        if not negated_here:
            # Attribution check: reporting a claim is not verifying it.
            attributed = any(marker in sentence_lower for marker in ATTRIBUTION_WORDS)
            if attributed:
                return StanceResult(
                    "Unclear",
                    best_sentence,
                    "Sumber melaporkan adanya klaim tersebut, bukan membenarkan isinya.",
                )
            return StanceResult(
                "Supports",
                best_sentence,
                "Kalimat sumber menyebut entitas dan predikat atom pada topik yang sama tanpa negasi.",
            )
    return StanceResult(
        "Unclear",
        best_sentence or None,
        "Tumpang tindih topik ada, tetapi aturan deterministik tidak dapat memutuskan arah hubungan.",
    )
