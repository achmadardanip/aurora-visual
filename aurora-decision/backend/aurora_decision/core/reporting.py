"""Deterministic, auditable decision report generator.

Every sentence in the report is grounded in bundle fields: verdicts cite
evidence ids/URLs with exact quotes, publishers, dates, independence groups.
Markdown/HTML export escapes all untrusted text; CSV escapes formulas.
"""

import html
import re
from datetime import datetime, timezone

FACT_LABELS_ID = {
    "Supported": "Didukung",
    "Contradicted": "Bertentangan",
    "InsufficientEvidence": "Bukti tidak cukup",
}
STANCE_ID = {
    "Supports": "Mendukung",
    "Contradicts": "Membantah",
    "NotRelevant": "Tidak relevan",
    "Unclear": "Tidak jelas",
}
CSV_FORMULA = re.compile(r"^[=+\-@]")


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def csv_cell(value) -> str:
    text = "" if value is None else str(value)
    if CSV_FORMULA.match(text):
        return "'" + text
    return text


def generate_markdown(bundle) -> str:
    """Full decision report as Markdown, grounded in the bundle."""
    decision = bundle.decision
    analysis = bundle.analysis
    retrieval = bundle.retrieval
    lines: list[str] = []
    lines.append(f"# Laporan Keputusan AURORA — Kasus {bundle.case_id}")
    lines.append("")
    lines.append(
        f"- Revisi klaim: {bundle.claim_revision} · Mode: {bundle.mode} · "
        f"Dibuat: {bundle.created_at.isoformat() if hasattr(bundle.created_at, 'isoformat') else bundle.created_at}"
    )
    if bundle.input.as_of:
        lines.append(f"- Batas waktu informasi (as_of): {bundle.input.as_of}")
    lines.append(f"- Caption: “{bundle.input.claim_text}”")
    verdict = FACT_LABELS_ID[decision.final_verdict]
    base = FACT_LABELS_ID[decision.base_label]
    lines.append(f"- Putusan operasional: **{verdict}** (label dasar model: {base})")
    calibration = decision.calibration
    lines.append(
        f"- Kalibrasi: {calibration.status}"
        + (f" · metode {calibration.method} · alpha {calibration.alpha}" if calibration.method else "")
        + (f" · {calibration.sample_count} unit kalibrasi independen" if calibration.sample_count else "")
    )
    if decision.abstention_flag:
        lines.append(f"- Alasan abstensi: {', '.join(decision.abstention_reasons) or '—'}")
    lines.append("")
    lines.append("## Ringkasan")
    lines.append(decision.decision_report.summary)
    if decision.decision_report.key_findings:
        lines.append("")
        lines.append("## Temuan utama")
        for finding in decision.decision_report.key_findings:
            lines.append(f"- {finding}")
    lines.append("")
    lines.append("## Keputusan per atom")
    lines.append("")
    lines.append("| Atom | Pernyataan | Label dasar | Status | Set keyakinan | Tautan bukti |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    evidence_map = {e.evidence_id: e for e in (retrieval.evidence_list if retrieval else [])}
    for verdict_item in decision.atomic_verdicts:
        atom = next((a for a in analysis.atomic_claims if a.atom_id == verdict_item.atom_id), None)
        statement = atom.statement if atom else "?"
        links = (
            ", ".join(
                f"{link.evidence_id} ({STANCE_ID[link.stance]})" for link in verdict_item.evidence_links
            )
            or "—"
        )
        conf = (
            ", ".join(verdict_item.confidence_set)
            if verdict_item.confidence_set is not None
            else "tidak terkalibrasi"
        )
        lines.append(
            f"| {verdict_item.atom_id} | {statement.replace('|', '/')} | "
            f"{FACT_LABELS_ID[verdict_item.base_label]} | {FACT_LABELS_ID[verdict_item.status]} | {conf} | {links} |"
        )
    lines.append("")
    cited = {link.evidence_id for v in decision.atomic_verdicts for link in v.evidence_links}
    if cited:
        lines.append("## Bukti yang dirujuk")
        for evidence_id in sorted(cited):
            item = evidence_map.get(evidence_id)
            if not item:
                continue
            url = f" ([sumber]({item.source.url}))" if item.source.url else ""
            lines.append(
                f"- **{evidence_id}** · {item.source.kind} · {item.source.publisher or 'penerbit tidak diketahui'} · "
                f"terbit {item.source.published_at or 'tidak diketahui'}{url} · kelompok independen "
                f"`{item.provenance.independence_group_id}`"
            )
            for verdict_item in decision.atomic_verdicts:
                for link in verdict_item.evidence_links:
                    if link.evidence_id == evidence_id and link.quote:
                        lines.append(
                            f"  - Kutipan ({verdict_item.atom_id}, {STANCE_ID[link.stance]}): “{link.quote}”"
                        )
    signals = retrieval.forensic_signals if retrieval else []
    if signals:
        lines.append("")
        lines.append("## Indikasi konten AI (informasi forensik, bukan putusan kebenaran)")
        for signal in signals:
            target = signal.target.kind
            score = signal.ai_generated_score
            lines.append(
                f"- {signal.provider} · target {target} · status {signal.status} · "
                f"indikasi AI {score if score is not None else 'tidak dinilai'}"
            )
    if decision.decision_report.unresolved_questions:
        lines.append("")
        lines.append("## Pertanyaan yang belum terjawab")
        for question in decision.decision_report.unresolved_questions:
            lines.append(f"- {question}")
    if decision.decision_report.limitations:
        lines.append("")
        lines.append("## Keterbatasan")
        for limitation in decision.decision_report.limitations:
            lines.append(f"- {limitation}")
    if decision.decision_report.suggested_next_steps:
        lines.append("")
        lines.append("## Langkah pemeriksaan berikutnya")
        for step in decision.decision_report.suggested_next_steps:
            lines.append(f"- {step}")
    if decision.human_review:
        review = decision.human_review
        lines.append("")
        lines.append("## Tinjauan manusia")
        lines.append(
            f"- {review.reviewer} pada {review.reviewed_at}: {FACT_LABELS_ID[review.verdict]} — {review.reason}"
        )
    lines.append("")
    lines.append("---")
    lines.append(
        "Putusan Didukung berarti klaim yang diperiksa didukung bukti layak pada run ini; "
        "bukan sertifikat keaslian seluruh gambar. Indikasi konten AI tidak membuat klaim otomatis palsu."
    )
    return "\n".join(lines)


def markdown_to_html(markdown_text: str) -> str:
    """Minimal, safe HTML rendering of the generated report (no raw HTML passthrough)."""
    out: list[str] = []
    in_list, in_table = False, False

    def close():
        nonlocal in_list, in_table
        if in_list:
            out.append("</ul>")
            in_list = False
        if in_table:
            out.append("</tbody></table>")
            in_table = False

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("| ") and "|" in stripped[2:]:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not in_table and cells and set(cells[0]) <= {"-", " ", ":"}:
                continue
            if not in_table:
                close()
                out.append('<table class="report-table"><tbody>')
                in_table = True
            out.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table and not stripped.startswith("|"):
            close()
        if stripped.startswith("- "):
            if not in_list:
                close()
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{render_inline(stripped[2:])}</li>")
        elif stripped.startswith("## "):
            close()
            out.append(f"<h2>{render_inline(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            close()
            out.append(f"<h1>{render_inline(stripped[2:])}</h1>")
        elif stripped == "---":
            close()
            out.append("<hr />")
        elif stripped:
            close()
            out.append(f"<p>{render_inline(stripped)}</p>")
    close()
    return (
        '<!doctype html><html lang="id"><head><meta charset="utf-8">'
        "<title>Laporan Keputusan AURORA</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;"
        "color:#16213e;line-height:1.55}table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "td,th{border:1px solid #cbd5e1;padding:6px 10px;text-align:left;font-size:0.9rem}"
        "blockquote{border-left:3px solid #2563eb;margin:0.4rem 0;padding-left:0.8rem;color:#334155}"
        "code{background:#eef2ff;padding:1px 4px;border-radius:3px;font-size:0.85em}hr{border:none;"
        "border-top:1px solid #cbd5e1;margin:1.5rem 0}</style></head><body>" + "".join(out) + "</body></html>"
    )


def render_inline(text: str) -> str:
    """Escape everything, then restore the two allowed inline markers."""
    safe = esc(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"`(.+?)`", r"<code>\1</code>", safe)
    safe = re.sub(r"“(.+?)”", r"<blockquote>“\1”</blockquote>", safe)
    return safe


def decision_csv(bundle) -> str:
    decision = bundle.decision
    rows = [["atom_id", "base_label", "status", "abstention", "reasons", "confidence_set", "links"]]
    for verdict_item in decision.atomic_verdicts:
        rows.append(
            [
                verdict_item.atom_id,
                verdict_item.base_label,
                verdict_item.status,
                str(verdict_item.abstention_flag),
                ";".join(verdict_item.abstention_reasons),
                ",".join(verdict_item.confidence_set) if verdict_item.confidence_set is not None else "",
                ";".join(f"{link.evidence_id}:{link.stance}" for link in verdict_item.evidence_links),
            ]
        )
    rows.append(
        [
            "CLAIM",
            decision.base_label,
            decision.final_verdict,
            str(decision.abstention_flag),
            ";".join(decision.abstention_reasons),
            ",".join(decision.confidence_set) if decision.confidence_set is not None else "",
            "",
        ]
    )
    return "\n".join(",".join(csv_cell(cell) for cell in row) for row in rows) + "\n"


def now_iso():
    return datetime.now(timezone.utc).isoformat()
