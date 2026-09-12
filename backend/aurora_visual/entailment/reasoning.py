import re

from app.models.contract import VisualAssessment

from aurora_visual.vision.features import ALIASES, COLOR_NAMES


def assess(atoms, features, transport, ocr, fixture=False):
    """Real color evidence is deliberately restricted to an unambiguous flat field.

    No object recognition, counting, affiliation, place or date inference is claimed.
    Similarity and absent evidence never establish contradiction.
    """
    result = []
    for i, atom in enumerate(atoms):
        status, support, contra, counter = "Unobservable", [], [], None
        rationale = "Bukti visual eksplisit belum cukup untuk menilai proposisi ini."
        observable = None
        words = set(re.findall(r"\w+", atom.statement.lower()))
        words |= {ALIASES[w] for w in list(words) if w in ALIASES}
        colors = [c for c in COLOR_NAMES if c in words]
        field_scope = bool(
            re.fullmatch(
                r"(bidang|latar|gambar|kanvas|field|background|image|canvas)(\s+(ini|itu|this))?(\s+(tidak|bukan|not))?",
                (atom.subject or "").lower().strip(),
            )
        )
        if atom.role == "attribute" and field_scope and len(colors) == 1:
            hist = features["color"][:, :6]
            dominant = hist.mean(0).argmax()
            purity = float(hist[:, dominant].min())
            if purity >= 0.96:
                observed, claimed = COLOR_NAMES[dominant], colors[0]
                consistent = (observed == claimed) != atom.qualifiers.negated
                status = "Supported" if consistent else "Contradicted"
                observable = purity
                # Every grid must show the same color: entire image is relevant coverage.
                regions = features["regions"]
                if consistent:
                    support = regions
                    rationale = f"Seluruh grid menunjukkan bidang {observed} (kemurnian minimum {purity:.2f}); lingkup klaim hanya warna bidang."
                else:
                    contra = regions
                    counter = f"Bidang yang diamati berwarna {observed}; klaim menyebut {'bukan ' if atom.qualifiers.negated else ''}{claimed}."
                    rationale = "Bantahan memakai warna alternatif yang terukur pada seluruh bidang."
        if atom.role in ("actor", "time", "location", "cause", "quantity", "relation"):
            rationale = {
                "time": "Tanggal tulisan/OCR bukan bukti tanggal pengambilan foto.",
                "location": "Lokasi memerlukan cue spesifik; pemandangan generik tidak cukup.",
                "actor": "Wajah atau kerumunan tidak membuktikan nama maupun afiliasi.",
                "cause": "Motif dan sebab tidak dapat ditetapkan dari tampilan semata.",
                "quantity": "Cakupan/oklusi dan detektor jumlah belum memadai untuk menilai kuantitas.",
                "relation": "Arah peran dipertahankan parser; relasi visual belum diamati secara eksplisit.",
            }[atom.role]
        result.append(
            VisualAssessment(
                atom_id=atom.atom_id,
                visual_status=status,
                probabilities=None,
                unmatched_mass=float(transport.unmatched_mass[i].detach()),
                observability_score=observable,
                supporting_regions=support,
                contradicting_regions=contra,
                counter_evidence=counter,
                rationale=("Fixture demonstrasi: " if fixture else "") + rationale,
                inference_kind="fixture" if fixture else "heuristic",
            )
        )
    return result
