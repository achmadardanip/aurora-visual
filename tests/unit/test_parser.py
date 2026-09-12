import pytest
from aurora_visual.atomization.parser import RuleAtomizer, StructuredLLMAtomizer
from aurora_visual.training.weak import perturb


def test_empty():
    with pytest.raises(ValueError):
        RuleAtomizer().parse("  ")


def test_negation_and_number():
    atoms = RuleAtomizer().parse("Dua orang tidak membawa 3 tas").atoms
    assert any(a.qualifiers.negated for a in atoms)
    assert any(a.role == "quantity" and a.qualifiers.quantity == 2 for a in atoms)
    assert all("3" in a.statement for a in atoms)


def test_role_swap():
    a = RuleAtomizer().parse("A mendorong B").atoms[0]
    b = RuleAtomizer().parse("B mendorong A").atoms[0]
    assert (a.subject, a.predicate, a.object) == ("A", "mendorong", "B")
    assert (b.subject, b.object) == ("B", "A")


def test_id_dates_location_and_affiliation():
    caption = "Mahasiswa melakukan aksi di Monas pada September 2026"
    result = RuleAtomizer().parse(caption)
    assert {"action", "actor", "location", "time"} <= {a.role for a in result.atoms}
    assert next(a for a in result.atoms if a.role == "time").qualifiers.time == "September 2026"
    assert next(a for a in result.atoms if a.role == "location").qualifiers.location == "Monas"
    assert all(a.parser_confidence is None for a in result.atoms)
    for a in result.atoms:
        for span in a.spans:
            assert caption[span.start : span.end]


def test_unicode_spans():
    caption = "📷 A mendorong B di Jakarta"
    atom = next(a for a in RuleAtomizer().parse(caption).atoms if a.role == "location")
    assert caption[atom.spans[0].start : atom.spans[0].end] == "di Jakarta"


def test_english_and_ambiguity():
    assert RuleAtomizer().parse("Alice pushes Bob", "en").atoms[0].subject == "Alice"
    result = RuleAtomizer().parse("Mereka di Jakarta")
    assert any(w.code == "COREFERENCE_UNRESOLVED" for w in result.warnings)


def test_unknown_llm_not_requested(monkeypatch):
    monkeypatch.setenv("AURORA_LLM_URL", "http://169.254.169.254")
    with pytest.raises(ValueError, match="allowlisted"):
        StructuredLLMAtomizer().parse("A mendorong B")


def test_weak_no_visual_label_for_unobserved_changes():
    pairs = perturb("A mendorong B di Monas pada 2026", source_label="Supported")
    assert any(p["kind"] == "role_swap" for p in pairs)
    assert all(p["label"] is None for p in pairs if p["kind"] in ("negation", "role_swap", "entity_swap"))
    assert any(p["kind"] == "voice" and p["label"] == "Supported" for p in pairs)
    assert all(p["confidence"] is None for p in pairs)


def test_parser_determinism_no_duplicates():
    parser = RuleAtomizer()
    caption = "Bidang ini berwarna merah. Bidang ini berwarna merah."
    a = parser.parse(caption).atoms
    b = parser.parse(caption).atoms
    assert a == b and len(a) == 1


def test_decimal_sign_passive_and_independent_clauses():
    parser = RuleAtomizer()
    atoms = parser.parse("A membawa 2.5 kg beras").atoms
    assert any(a.qualifiers.quantity == 2.5 for a in atoms)
    assert any(a.qualifiers.quantity == -3 for a in parser.parse("A membawa -3 kg").atoms)
    assert any(w.code == "QUANTITY_AMBIGUOUS" for w in parser.parse("A membawa 1.000 kg").warnings)
    relation = parser.parse("Bob is pushed by Alice", "en").atoms[0]
    assert (relation.subject, relation.predicate, relation.object) == ("Alice", "pushes", "Bob")
    atoms = parser.parse("A mendorong B dan C menarik D").atoms
    assert [(a.subject, a.object) for a in atoms] == [("A", "B"), ("C", "D")]
