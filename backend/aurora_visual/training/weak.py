"""Typed matched perturbations; lexical changes do not imply visual contradiction."""

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from aurora_visual.atomization.parser import RuleAtomizer


@dataclass
class Pair:
    source: str
    modified: str
    kind: str
    label: str | None
    provenance: str
    confidence: float | None
    quality: dict
    atom_correspondence: list[tuple[int, int]]


def perturb(caption, explicit_visual_roles=(), source_label=None):
    parser = RuleAtomizer()
    source = parser.parse(caption).atoms
    proposed = []
    for a in source:
        if a.subject and a.object and a.role == "relation":
            replacement = f"{a.object} {a.predicate} {a.subject}"
            proposed.append((caption.replace(a.statement, replacement, 1), "role_swap", "relation", False))
        if a.subject and re.search(r"\b(A|B|Andi|Budi|Alice|Bob)\b", a.subject):
            proposed.append((caption.replace(a.subject, "Citra", 1), "entity_swap", "actor", False))
    for old, new, kind, role in [
        ("merah", "biru", "attribute_swap", "attribute"),
        ("di atas", "di bawah", "relation_swap", "relation"),
        ("dua", "tiga", "number_change", "quantity"),
        ("2", "3", "number_change", "quantity"),
    ]:
        if re.search(r"\b" + re.escape(old) + r"\b", caption):
            proposed.append(
                (re.sub(r"\b" + re.escape(old) + r"\b", new, caption, count=1), kind, role, False)
            )
    if source:
        a = source[0]
        if a.predicate in caption:
            changed = (
                caption.replace("tidak ", "", 1)
                if a.qualifiers.negated
                else caption.replace(a.predicate, "tidak " + a.predicate, 1)
            )
            proposed.append((changed, "negation", a.role, False))
    for old, new in [("memakai", "mengenakan"), ("melakukan aksi", "menggelar aksi"), ("bidang", "latar")]:
        if old in caption.lower():
            proposed.append((re.sub(old, new, caption, count=1, flags=re.I), "synonym", "action", True))
    for a in source:
        if a.predicate == "mendorong" and a.subject and a.object:
            proposed.append(
                (
                    caption.replace(a.statement, f"{a.object} didorong oleh {a.subject}", 1),
                    "voice",
                    "relation",
                    True,
                )
            )
    # A minimal punctuation paraphrase leaves semantics invariant.
    proposed.append((caption.rstrip(".") + ".", "paraphrase", "action", True))
    pairs = []
    for modified, kind, role, positive in proposed:
        if modified == caption:
            continue
        parsed = parser.parse(modified).atoms
        token_change = 1 - SequenceMatcher(None, caption.split(), modified.split()).ratio()
        grammatical = bool(parsed) and all(a.predicate and a.statement.strip() for a in parsed)
        minimal = token_change <= 0.65 or kind in ("voice", "role_swap")
        visual = role in explicit_visual_roles
        # Align by role and semantic SPO, including canonical active/passive for voice.
        correspondence = []
        used = set()
        for i, a in enumerate(source):
            candidates = [j for j, b in enumerate(parsed) if b.role == a.role and j not in used]
            if candidates:
                j = max(
                    candidates, key=lambda j: SequenceMatcher(None, a.statement, parsed[j].statement).ratio()
                )
                correspondence.append((i, j))
                used.add(j)
        if grammatical and minimal:
            pairs.append(
                asdict(
                    Pair(
                        caption,
                        modified,
                        kind,
                        source_label
                        if positive
                        else "Contradicted"
                        if visual and source_label == "Supported"
                        else None,
                        "rule-generated; conditional on supported source and explicit alternative annotation",
                        None,
                        {
                            "token_edit_fraction": token_change,
                            "grammar_rule_pass": grammatical,
                            "minimality_pass": minimal,
                            "visual_role_annotated": visual,
                            "empirical_accuracy": None,
                        },
                        correspondence,
                    )
                )
            )
    return pairs
