import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowRight, Check, ChevronRight, CircleHelp, Clock3, Download, FileJson,
  FlaskConical, LoaderCircle, Plus, ScanLine, Settings2, ShieldCheck, Upload, X,
} from "lucide-react";

type MediaRef = {
  asset_id: string; sha256: string; media_type: string;
  width: number; height: number; uri: string;
};
type Evidence = {
  evidence_id: string;
  source: { kind: string; url: string | null; title: string; publisher: string | null; published_at: string | null };
  content: { text: string; excerpt: string; status: string };
  provenance: { independence_group_id: string; temporal_eligible: boolean | null };
};
type EvidenceLink = {
  evidence_id: string; atom_id: string; stance: string;
  quote: string | null; rationale: string;
};
type AtomicDecision = {
  atom_id: string; base_label: string; status: string;
  probabilities: Record<string, number> | null;
  confidence_set: string[] | null;
  abstention_flag: boolean; abstention_reasons: string[];
  evidence_links: EvidenceLink[]; visual_atom_ids: string[];
  explanation: string;
  calibration: { status: string; method: string | null; alpha: number | null; sample_count: number };
};
type Atom = {
  atom_id: string; statement: string; role: string;
  qualifiers: { negated: boolean; quantity: number | null; time: string | null; location: string | null };
};
type Decision = {
  run: { run_id: string; status: string; warnings: { code: string; message: string }[] };
  atom_set_id: string;
  atomic_verdicts: AtomicDecision[];
  base_label: string; final_verdict: string;
  abstention_flag: boolean; abstention_reasons: string[];
  calibration: { status: string; method: string | null; alpha: number | null; sample_count: number };
  decision_report: {
    summary: string; key_findings: string[]; unresolved_questions: string[];
    limitations: string[]; suggested_next_steps: string[];
  };
  human_review: { reviewer: string; reviewed_at: string; verdict: string; reason: string } | null;
};
type Bundle = {
  schema_version: string; case_id: string; claim_revision: number; mode: string;
  input: { claim_text: string; language: string; images: MediaRef[]; as_of: string | null };
  analysis: { atom_set_id: string; atomic_claims: Atom[] } | null;
  retrieval: { evidence_list: Evidence[]; forensic_signals: unknown[] } | null;
  decision: Decision | null;
  warnings: { code: string; message: string }[];
  extensions: Record<string, any>;
};
type Job = {
  job_id: string; status: string; progress: string;
  result: Bundle | null; error: { message: string; detail?: string } | null;
};
type Capability = { provider: string; mode: string; capability: string; status: string; message: string | null };

const FACT_ID: Record<string, string> = {
  Supported: "Didukung", Contradicted: "Bertentangan", InsufficientEvidence: "Bukti tidak cukup",
};
const STANCE_ID: Record<string, string> = {
  Supports: "Mendukung", Contradicts: "Membantah", NotRelevant: "Tidak relevan", Unclear: "Tidak jelas",
};
const ROLE_ID: Record<string, string> = {
  actor: "Pelaku", action: "Aksi", object: "Objek", attribute: "Atribut",
  location: "Lokasi", time: "Waktu", quantity: "Jumlah", relation: "Relasi", cause: "Sebab",
};
const statusLabel = (s: string) => s.replaceAll("_", " ");

function api<T>(path: string, init?: RequestInit): Promise<T> {
  return fetch(path, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...(init?.headers || {}),
    },
  }).then(async (r) => {
    if (!r.ok) {
      const body = await r.json().catch(() => ({ error: { message: "Permintaan gagal" } }));
      throw new Error(body.error?.message || `HTTP ${r.status}`);
    }
    return r.json();
  });
}

export default function App() {
  const [page, setPage] = useState<"fuse" | "history" | "settings">("fuse");
  const [caption, setCaption] = useState("");
  const [language, setLanguage] = useState("id");
  const [media, setMedia] = useState<MediaRef[]>([]);
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState<Job | null>(null);
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [selectedAtom, setSelectedAtom] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [capabilities, setCapabilities] = useState<Capability[]>([]);
  const [calibration, setCalibration] = useState<{ artifacts: any[]; alpha: number } | null>(null);
  const [history, setHistory] = useState<{ case_id: string; caption: string; mode: string; updated_at: number; status: string }[]>([]);
  const [reviewer, setReviewer] = useState("");
  const [reviewVerdict, setReviewVerdict] = useState("InsufficientEvidence");
  const [reviewReason, setReviewReason] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const importInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try { setCapabilities((await api<{ capabilities: Capability[] }>("/ready")).capabilities || []); } catch { /* ok */ }
    try { setCalibration(await api("/api/v1/calibration")); } catch { /* ok */ }
    try { setHistory(await api("/api/v1/cases")); } catch { /* ok */ }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    if (!running) return;
    let active = true;
    const poll = async () => {
      try {
        const next = await api<Job>(`/api/v1/jobs/${running.job_id}`);
        if (!active) return;
        setRunning(next);
        if (!["queued", "running"].includes(next.status)) {
          setRunning(null);
          if (next.result) {
            setBundle(next.result);
            setSelectedAtom(next.result.decision?.atomic_verdicts[0]?.atom_id || null);
            setNotice(next.status === "partial" ? "Fusi selesai dengan peringatan." : "Fusi selesai.");
          }
          if (next.error) setError(next.error.detail ? `${next.error.message} (${next.error.detail})` : next.error.message);
          void refresh();
        }
      } catch (e) { if (active) setError((e as Error).message); }
    };
    void poll();
    const timer = setInterval(() => void poll(), 1000);
    return () => { active = false; clearInterval(timer); };
  }, [running, refresh]);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("images", f));
    try {
      const result = await api<{ images: MediaRef[] }>("/api/v1/media", { method: "POST", body: form });
      setMedia((prev) => [...prev, ...result.images].slice(0, 8));
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  async function submit(endpoint: "fuse" | "pipeline") {
    if (!caption.trim() || !bundle?.analysis) return;
    setBusy(true); setError(""); setNotice("");
    try {
      const job = await api<Job>(`/api/v1/${endpoint}`, {
        method: "POST",
        headers: { "Idempotency-Key": crypto.randomUUID() },
        body: JSON.stringify(bundle),
      });
      setRunning({ ...job, progress: "Menunggu giliran", result: null, error: null });
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  async function loadDemo() {
    setError(""); setNotice("");
    try {
      const b = await api<Bundle>("/api/v1/demo/fusion");
      setBundle(b); setCaption(b.input.claim_text); setSelectedAtom(b.analysis?.atomic_claims[0]?.atom_id || null);
      setNotice("Fixture demo dimuat: 2 atom, 3 bukti (1 duplikat sindikasi). Jalankan fusi.");
    } catch (e) { setError((e as Error).message); }
  }

  async function importBundle(file: File | undefined) {
    if (!file) return;
    setBusy(true); setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const result = await api<{ bundle: Bundle }>("/api/v1/import", { method: "POST", body: form });
      setBundle(result.bundle); setCaption(result.bundle.input.claim_text);
      setSelectedAtom(result.bundle.analysis?.atomic_claims[0]?.atom_id || null);
      setNotice(result.bundle.analysis ? "Bundle diimpor. Jalankan fusi." : "Bundle diimpor tanpa analysis; jalur pipeline akan memanggil layanan 1.");
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  async function loadCase(caseId: string) {
    setError("");
    try {
      const b = await api<Bundle>(`/api/v1/cases/${caseId}`);
      setBundle(b); setCaption(b.input.claim_text); setSelectedAtom(null); setPage("fuse");
    } catch (e) { setError((e as Error).message); }
  }

  async function submitReview() {
    if (!bundle || !reviewReason.trim()) return;
    setBusy(true); setError("");
    try {
      const b = await api<Bundle>(`/api/v1/cases/${bundle.case_id}/review`, {
        method: "POST",
        body: JSON.stringify({ reviewer, verdict: reviewVerdict, reason: reviewReason }),
      });
      setBundle(b);
      setReviewReason("");
      setNotice("Tinjauan manusia tersimpan; hasil model tetap utuh untuk audit.");
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  const decision = bundle?.decision;
  const atoms = bundle?.analysis?.atomic_claims || [];
  const evidenceMap = new Map((bundle?.retrieval?.evidence_list || []).map((e) => [e.evidence_id, e]));
  const selected = decision?.atomic_verdicts.find((v) => v.atom_id === selectedAtom);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">A3</span>
          <div><strong>AURORA Decision</strong><small>Fusi bukti &amp; keputusan</small></div>
        </div>
        <nav>
          {([["fuse", "Fusi & keputusan", ScanLine], ["history", "Riwayat kasus", FileJson], ["settings", "Kalibrasi & layanan", Settings2]] as const).map(
            ([key, label, Icon]) => (
              <button key={key} className={page === key ? "active" : ""} onClick={() => { setPage(key); if (key !== "fuse") void refresh(); }}>
                <Icon size={16} /> {label}
              </button>
            ),
          )}
        </nav>
        <div className="sidebar-note">
          <ShieldCheck size={21} />
          <strong>Konservatif saat ragu.</strong>
          <p>Tanpa kalibrasi yang cocok, putusan operasional adalah Bukti tidak cukup. Tinjauan manusia tercatat terpisah.</p>
        </div>
        <div className="sidebar-footer"><span className="online-dot" /> AURORA v1.0 <span>v1.0</span></div>
      </aside>
      <main>
        <header className="topbar">
          <div>AURORA <ChevronRight size={14} /> <strong>{page === "fuse" ? "Fusi bukti" : page === "history" ? "Riwayat kasus" : "Kalibrasi & layanan"}</strong></div>
        </header>
        <div className="main-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow"><span /> FUSI MULTI-BUKTI &amp; CONFORMAL ABSTENTION</div>
              <h1>{page === "fuse" ? "Satu putusan, jejak lengkap." : page === "history" ? "Setiap putusan tersimpan." : "Kalibrasi dan layanan."}</h1>
              <p>{page === "fuse"
                ? "Bukti dari layanan 2 diverifikasi per atom, difusi dengan pengelompokan sumber independen, lalu dikalibrasi."
                : page === "history" ? "Kasus, putusan, dan tinjauan manusia tersimpan di workspace Anda."
                : "Periksa artefak kalibrasi, alpha, dan status layanan upstream."}</p>
            </div>
          </div>
          {error && <div className="banner error" role="alert"><CircleHelp size={18} /><span>{error}</span><button aria-label="Tutup error" onClick={() => setError("")}><X size={16} /></button></div>}
          {notice && <div className="banner notice" role="status"><Check size={18} /><span>{notice}</span></div>}

          {page === "fuse" && (
            <>
              <section className="card input-card">
                <div className="card-top">
                  <div><span className="section-number">MASUKAN</span><h2>Bundle untuk difusi</h2></div>
                </div>
                <p className="evidence-meta">
                  Impor bundle dari layanan 1/2 (JSON/ZIP) atau muat fixture demo. Jalur pipeline memanggil layanan 1 dan 2 otomatis bila bundle belum memiliki analysis/retrieval.
                </p>
                <div className="field-row">
                  <input ref={importInput} type="file" accept=".json,.zip" className="sr-only" aria-label="Impor berkas hasil"
                    onChange={(e) => { void importBundle(e.target.files?.[0]); e.target.value = ""; }} />
                  <button className="button secondary" onClick={() => importInput.current?.click()}><Upload size={14} /> Impor bundle</button>
                  <button className="button secondary" onClick={() => void loadDemo()}><FlaskConical size={14} /> Muat fixture demo</button>
                </div>
                <label className="field-label" htmlFor="caption">CAPTION</label>
                <textarea id="caption" value={caption} rows={2} readOnly />
                <div className="input-footer">
                  <span><ShieldCheck size={15} /> {bundle?.analysis ? `Analysis termuat · ${atoms.length} atom` : "Belum ada analysis"}</span>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button className="button secondary" onClick={() => void submit("pipeline")} disabled={busy || !!running || !caption.trim()}>
                      <LoaderCircle size={15} /> Pipeline penuh
                    </button>
                    <button className="button primary" onClick={() => void submit("fuse")} disabled={busy || !!running || !bundle?.analysis}>
                      {busy || running ? <LoaderCircle className="spin" size={17} /> : <ScanLine size={17} />}
                      {running ? "Menggabungkan…" : "Jalankan fusi"} <ArrowRight size={16} />
                    </button>
                  </div>
                </div>
              </section>

              {running && (
                <div className="progress-card" role="status">
                  <LoaderCircle className="spin" size={22} />
                  <div><strong>{running.progress}</strong><p>Pipeline dapat memanggil layanan upstream; halaman boleh dimuat ulang.</p></div>
                </div>
              )}

              {decision && (
                <section className="card result-card">
                  <div className="card-top">
                    <div>
                      <span className="section-number">PUTUSAN</span>
                      <h2>{FACT_ID[decision.final_verdict]}{decision.abstention_flag ? " (abstain)" : ""}</h2>
                    </div>
                    <div className="export-buttons">
                      {(["md", "html", "csv"] as const).map((f) => (
                        <button key={f} onClick={() => window.open(`/api/v1/cases/${bundle!.case_id}/report?format=${f === "md" ? "md" : f}`, "_blank")}>
                          <Download size={14} /> {f.toUpperCase()}
                        </button>
                      ))}
                      <button onClick={() => window.open(`/api/v1/cases/${bundle!.case_id}/export?format=zip`, "_blank")}>
                        <Download size={14} /> ZIP
                      </button>
                    </div>
                  </div>
                  <p>{decision.decision_report.summary}</p>
                  {decision.abstention_flag && (
                    <div className="banner notice" role="note">
                      <CircleHelp size={16} />
                      <span>Alasan abstensi: {decision.abstention_reasons.join(", ")}. Label dasar model tetap {FACT_ID[decision.base_label]} untuk audit.</span>
                    </div>
                  )}
                  <dl className="evidence-scores">
                    <div><dt>Status kalibrasi</dt><dd>{decision.calibration.status === "uncalibrated" ? "Belum terkalibrasi" : decision.calibration.status}{decision.calibration.alpha != null ? ` · alpha ${decision.calibration.alpha}` : ""}</dd></div>
                    <div><dt>Rute fusi</dt><dd>{bundle!.extensions?.aurora_decision?.input_route || "—"}</dd></div>
                    <div><dt>Cakupan bobot dukungan</dt><dd>{Math.round((bundle!.extensions?.aurora_decision?.coverage ?? 0) * 100)}%</dd></div>
                    <div><dt>Peringatan</dt><dd>{decision.run.warnings.length}</dd></div>
                  </dl>

                  <h3>Matriks atom × bukti</h3>
                  <div className="atom-list">
                    {decision.atomic_verdicts.map((v) => {
                      const atom = atoms.find((a) => a.atom_id === v.atom_id);
                      return (
                        <button key={v.atom_id} className={`atom-row ${selectedAtom === v.atom_id ? "selected" : ""}`} onClick={() => setSelectedAtom(v.atom_id)}>
                          <span className="atom-number">{v.atom_id.slice(-2)}</span>
                          <span className="atom-content">
                            <span className="atom-role">{ROLE_ID[atom?.role || ""] || v.atom_id}</span>
                            <strong>{atom?.statement}</strong>
                            <span className={`status-badge ${v.status}`}>{FACT_ID[v.status]}{v.abstention_flag ? " · abstain" : ""}</span>
                          </span>
                          <ChevronRight size={16} />
                        </button>
                      );
                    })}
                  </div>

                  {selected && (
                    <div className="card inspection">
                      <div className="inspection-title">
                        <span className="tiny-tag">{selected.atom_id}</span>
                        <h3>Dasar penilaian</h3>
                        <span>{selected.explanation}</span>
                      </div>
                      <dl className="evidence-scores">
                        <div><dt>Label dasar</dt><dd>{FACT_ID[selected.base_label]}</dd></div>
                        <div><dt>Status operasional</dt><dd>{FACT_ID[selected.status]}</dd></div>
                        <div><dt>Set keyakinan</dt><dd>{selected.confidence_set === null ? "Tidak terkalibrasi" : selected.confidence_set.length ? selected.confidence_set.map((s) => FACT_ID[s]).join(", ") : "(himpunan kosong)"}</dd></div>
                        <div><dt>Kalibrasi atom</dt><dd>{selected.calibration.status}{selected.calibration.sample_count ? ` · ${selected.calibration.sample_count} unit` : ""}</dd></div>
                      </dl>
                      {selected.evidence_links.length > 0 && (
                        <>
                          <h4>Bukti terkait</h4>
                          {selected.evidence_links.map((link, i) => {
                            const item = evidenceMap.get(link.evidence_id);
                            return (
                              <blockquote key={i}>
                                “{link.quote || "(tanpa kutipan persis)"}”
                                <small>
                                  {STANCE_ID[link.stance]} · {item?.source.publisher || "?"} · {item?.source.published_at?.slice(0, 10) || "tanggal tidak diketahui"} ·
                                  kelompok independen {item?.provenance.independence_group_id.slice(0, 12)}…
                                  {item?.source.url && <> · <a href={item.source.url} target="_blank" rel="noreferrer noopener">sumber ↗</a></>}
                                </small>
                                <small>{link.rationale}</small>
                              </blockquote>
                            );
                          })}
                        </>
                      )}
                      {selected.abstention_reasons.length > 0 && (
                        <p className="evidence-meta">Alasan abstensi atom: {selected.abstention_reasons.join(", ")}.</p>
                      )}
                    </div>
                  )}

                  <h3>Tinjauan manusia</h3>
                  {decision.human_review ? (
                    <blockquote>
                      {decision.human_review.reviewer} · {FACT_ID[decision.human_review.verdict]}
                      <small>{decision.human_review.reason}</small>
                    </blockquote>
                  ) : (
                    <div className="field-row">
                      <label>Nama peninjau<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="opsional" /></label>
                      <label>Putusan
                        <select value={reviewVerdict} onChange={(e) => setReviewVerdict(e.target.value)}>
                          <option value="Supported">Didukung</option>
                          <option value="Contradicted">Bertentangan</option>
                          <option value="InsufficientEvidence">Bukti tidak cukup</option>
                        </select>
                      </label>
                      <label>Alasan<input value={reviewReason} onChange={(e) => setReviewReason(e.target.value)} placeholder="wajib diisi" /></label>
                      <button className="button secondary" onClick={() => void submitReview()} disabled={busy || !reviewReason.trim()}>
                        <Check size={14} /> Simpan tinjauan
                      </button>
                    </div>
                  )}
                  <details>
                    <summary>Pertanyaan belum terjawab &amp; keterbatasan</summary>
                    <ul>
                      {decision.decision_report.unresolved_questions.map((q, i) => <li key={i}>{q}</li>)}
                      {decision.decision_report.limitations.map((l, i) => <li key={`l${i}`}>{l}</li>)}
                    </ul>
                  </details>
                </section>
              )}
            </>
          )}

          {page === "history" && (
            <section className="card history-page">
              <div className="card-top"><h2>Semua kasus <span className="tiny-tag">{history.length}</span></h2></div>
              {history.map((h) => (
                <button className="history-row" key={h.case_id} onClick={() => void loadCase(h.case_id)}>
                  <span className="history-file"><FileJson size={22} /></span>
                  <span><strong>{h.caption}</strong><small>{new Date(h.updated_at * 1000).toLocaleString("id-ID")}</small></span>
                  <span className="tiny-tag">{FACT_ID[h.status] || h.status}</span>
                  <ChevronRight size={18} />
                </button>
              ))}
              {!history.length && <div className="empty-history"><Clock3 size={36} /><h3>Belum ada kasus</h3><p>Muat fixture demo lalu jalankan fusi.</p></div>}
            </section>
          )}

          {page === "settings" && (
            <section className="card settings-groups">
              <h3>Artefak kalibrasi</h3>
              <p className="evidence-meta">
                Alpha aktif: {calibration?.alpha ?? "—"} · {calibration?.artifacts.length ?? 0} artefak tersimpan.
                Mengubah alpha memerlukan artefak dengan quantile yang sah untuk alpha tersebut.
              </p>
              <div className="trace-list">
                {(calibration?.artifacts || []).map((a) => (
                  <div className="trace-item" key={a.calibration_id}>
                    <code>{a.calibration_id.slice(0, 20)}…</code>
                    <span>{a.method} · alpha {a.alpha} · rute {a.input_route} · {a.sample_count} unit</span>
                  </div>
                ))}
                {!calibration?.artifacts.length && <p className="empty-note">Belum terkalibrasi. Jalankan CLI calibrate untuk membuat artefak.</p>}
              </div>
              <h3>Status layanan</h3>
              <div className="capability-grid">
                {capabilities.map((c) => (
                  <div className="card capability-card" key={c.provider}>
                    <div><span className="tiny-tag">{c.mode}</span><span className={`cap-status ${c.status}`}>{statusLabel(c.status)}</span></div>
                    <h3>{c.provider}</h3><p>{c.message}</p><small>{c.capability}</small>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
