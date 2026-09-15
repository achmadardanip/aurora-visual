import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle, ArrowRight, Check, ChevronRight, CircleHelp, Clock3, Download,
  FileJson, FlaskConical, Image as ImageIcon, LoaderCircle, Plus, ScanLine,
  Search, Settings2, ShieldCheck, Upload, X,
} from "lucide-react";

const base = "";

type MediaRef = {
  asset_id: string; sha256: string; media_type: string;
  width: number; height: number; uri: string;
};
type Evidence = {
  evidence_id: string; atom_ids: string[];
  source: {
    provider: string; kind: string; url: string | null; title: string;
    publisher: string | null; language: string | null;
    published_at: string | null; retrieved_at: string;
  };
  content: { text: string; excerpt: string; sha256: string; status: string };
  provenance: {
    original_url: string | null; first_seen_at: string | null;
    discovery_method: string; duplicate_cluster_id: string;
    independence_group_id: string; temporal_eligible: boolean | null;
    usage_note: string | null;
    image_matches: { asset_id: string; matched_url: string | null; match_type: string; score: number | null }[];
  };
  relevance_score: number | null; credibility_score: number | null;
  rerank_score: number | null; score_breakdown: Record<string, number>;
};
type ForensicSignal = {
  signal_id: string;
  target: { kind: string; asset_id: string | null; text_sha256: string | null };
  modality: string; provider: string; model_version: string | null;
  status: string; raw_score: number | null;
  raw_scale: { min: number; max: number; higher_means_ai: boolean } | null;
  ai_generated_score: number | null; raw_label: string | null;
  assessment: string; calibration_status: string;
  applicable_language: string | null; limitations: string[];
  analyzed_at: string; error_code: string | null;
};
type Bundle = {
  schema_version: string; case_id: string; claim_revision: number; mode: string;
  input: { claim_text: string; language: string; images: MediaRef[]; as_of: string | null };
  analysis: null;
  retrieval: {
    run: { run_id: string; status: string; warnings: { code: string; message: string }[] };
    atom_set_id: string | null;
    evidence_list: Evidence[]; forensic_signals: ForensicSignal[];
    query_log: { query_id: string; provider: string; query: string; duration_ms: number; result_count: number }[];
    provider_status: { provider: string; capability: string; status: string; message: string | null }[];
  } | null;
  decision: null;
  warnings: { code: string; message: string }[];
  extensions: Record<string, unknown>;
};
type Job = {
  job_id: string; status: string; progress: string;
  result: Bundle | null; error: { message: string; detail?: string } | null;
};
type Capability = { provider: string; mode: string; capability: string; status: string; message: string | null };

const KIND_LABEL: Record<string, string> = {
  fact_check: "Cek fakta", news: "Berita", official: "Sumber resmi",
  web: "Web", image_provenance: "Asal gambar", local_corpus: "Korpus lokal",
  user_supplied: "Dokumen pengguna",
};
const SIGNAL_STATUS: Record<string, string> = {
  ok: "Berhasil", inconclusive: "Belum konklusif", unsupported: "Tidak didukung",
  unavailable: "Tidak tersedia", failed: "Gagal",
};
const ASSESSMENT_LABEL: Record<string, string> = {
  likely_ai_generated: "Indikasi konten AI",
  likely_human_or_camera: "Kemungkinan manusia/kamera",
  uncertain: "Tidak pasti", not_assessed: "Tidak dinilai",
};
const statusLabel = (s: string) =>
  SIGNAL_STATUS[s] || s.replaceAll("_", " ");
const fmtDate = (v: string | null) =>
  v ? new Date(v).toLocaleString("id-ID", { dateStyle: "medium", timeStyle: "short" }) : "Tidak diketahui";

function api<T>(path: string, init?: RequestInit): Promise<T> {
  return fetch(base + path, {
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
    return r.status === 204 ? null : r.json();
  });
}

export default function App() {
  const [page, setPage] = useState<"search" | "history" | "settings">("search");
  const [caption, setCaption] = useState("");
  const [language, setLanguage] = useState("id");
  const [asOf, setAsOf] = useState("");
  const [media, setMedia] = useState<MediaRef[]>([]);
  const [mode, setMode] = useState<"demo" | "live">("live");
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState<Job | null>(null);
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [tab, setTab] = useState<"evidence" | "image" | "ai" | "trace">("evidence");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [capabilities, setCapabilities] = useState<Capability[]>([]);
  const [kindFilter, setKindFilter] = useState("all");
  const [sort, setSort] = useState<"relevance" | "date">("relevance");
  const [history, setHistory] = useState<{ case_id: string; caption: string; mode: string; updated_at: number; status: string }[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const importInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const ready = await api<{ capabilities: Capability[] }>("/ready");
      setCapabilities(ready.capabilities || []);
    } catch { /* readiness not fatal for the UI */ }
    try {
      setHistory(await api<typeof history>("/api/v1/cases"));
    } catch { /* empty history is fine */ }
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
            setNotice(next.status === "partial" ? "Pencarian selesai dengan peringatan penyedia." : "Pencarian bukti selesai.");
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
    setBusy(true); setError("");
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("images", f));
    try {
      const result = await api<{ images: MediaRef[] }>("/api/v1/media", { method: "POST", body: form });
      setMedia((prev) => [...prev, ...result.images].slice(0, 8));
      setNotice(`${result.images.length} gambar siap dipakai.`);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  async function submit() {
    if (!caption.trim()) return;
    setBusy(true); setError(""); setNotice("");
    try {
      const payload = {
        schema_version: "1.0.0", case_id: crypto.randomUUID(), claim_revision: 1, mode,
        created_at: new Date().toISOString(),
        input: {
          claim_text: caption, language, images: media, as_of: asOf ? new Date(asOf).toISOString() : null,
        },
        analysis: null, retrieval: null, decision: null, warnings: [],
        extensions: {},
      };
      const job = await api<Job>("/api/v1/retrieve", {
        method: "POST",
        headers: { "Idempotency-Key": crypto.randomUUID() },
        body: JSON.stringify(payload),
      });
      setRunning({ ...job, progress: "Menunggu giliran", result: null, error: null });
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  async function loadDemo(fixture: string) {
    setError(""); setNotice("");
    try {
      const b = await api<Bundle>(`/api/v1/demo/${fixture}`);
      setMode("demo");
      setCaption(b.input.claim_text);
      setMedia([]);
      setNotice("Fixture demo dimuat. Jalankan pencarian untuk melihat hasil sintetis.");
    } catch (e) { setError((e as Error).message); }
  }

  async function importBundle(file: File | undefined) {
    if (!file) return;
    setBusy(true); setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const result = await api<{ bundle: Bundle }>("/api/v1/import", { method: "POST", body: form });
      setBundle(result.bundle);
      setCaption(result.bundle.input.claim_text);
      setMode(result.bundle.mode as "demo" | "live");
      setNotice("Bundle diimpor. Jalankan pencarian ulang untuk bukti baru.");
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }

  async function loadCase(caseId: string) {
    setError("");
    try {
      const b = await api<Bundle>(`/api/v1/cases/${caseId}`);
      setBundle(b); setCaption(b.input.claim_text); setMode(b.mode as "demo" | "live"); setMedia(b.input.images);
      setPage("search");
    } catch (e) { setError((e as Error).message); }
  }

  const evidence = bundle?.retrieval?.evidence_list || [];
  const signals = bundle?.retrieval?.forensic_signals || [];
  const imageSignals = signals.filter((s) => s.modality === "image");
  const textSignals = signals.filter((s) => s.modality === "text");
  const imageMatches = evidence.flatMap((e) =>
    e.provenance.image_matches.map((m) => ({ ...m, evidence: e })));
  const shown = evidence
    .filter((e) => kindFilter === "all" || e.source.kind === kindFilter)
    .sort((a, b) =>
      sort === "relevance"
        ? (b.relevance_score ?? 0) - (a.relevance_score ?? 0)
        : (b.source.published_at || "").localeCompare(a.source.published_at || ""));

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">A2</span>
          <div>
            <strong>AURORA Evidence</strong>
            <small>Pencarian &amp; deteksi bukti</small>
          </div>
        </div>
        <nav>
          {([["search", "Pencarian bukti", ScanLine], ["history", "Riwayat kasus", FileJson], ["settings", "Pengaturan", Settings2]] as const).map(
            ([key, label, Icon]) => (
              <button key={key} className={page === key ? "active" : ""} onClick={() => { setPage(key); if (key !== "search") void refresh(); }}>
                <Icon size={16} /> {label}
              </button>
            ),
          )}
        </nav>
        <div className="sidebar-note">
          <ShieldCheck size={21} />
          <strong>Bukti, bukan putusan.</strong>
          <p>Hasil pencarian adalah bahan tinjauan manusia. Indikasi konten AI bukan berarti klaim palsu.</p>
        </div>
        <div className="sidebar-footer"><span className="online-dot" /> {mode === "live" ? "Korpus lokal + provider aktif" : "Demo sintetis"} <span>v1.0</span></div>
      </aside>
      <main>
        <header className="topbar">
          <div>AURORA <ChevronRight size={14} /> <strong>{page === "search" ? "Pencarian bukti" : page === "history" ? "Riwayat kasus" : "Pengaturan penyedia"}</strong></div>
        </header>
        <div className="main-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow"><span /> PENCARIAN BUKTI MULTISUMBER</div>
              <h1>{page === "search" ? "Cari bukti untuk satu klaim." : page === "history" ? "Jejak setiap pencarian." : "Penyedia yang dapat diganti."}</h1>
              <p>{page === "search"
                ? "Klaim dicari ke korpus cek fakta lokal dan penyedia web aktif; hasil didedup dan diberi peringkat."
                : page === "history" ? "Kasus dan hasil pencarian tersimpan di workspace Anda."
                : "Ganti penyedia pencarian dan deteksi AI lewat konfigurasi server, tanpa mengubah logika."}</p>
            </div>
          </div>
          {error && <div className="banner error" role="alert"><CircleHelp size={18} /><span>{error}</span><button aria-label="Tutup error" onClick={() => setError("")}><X size={16} /></button></div>}
          {notice && <div className="banner notice" role="status"><Check size={18} /><span>{notice}</span></div>}

          {page === "search" && (
            <>
              <section className="card input-card">
                <div className="card-top">
                  <div><span className="section-number">MASUKAN</span><h2>Klaim yang akan dicari</h2></div>
                  <div className="mode-switch" aria-label="Mode pencarian">
                    <button disabled={!!running} className={mode === "demo" ? "active" : ""} onClick={() => setMode("demo")}><FlaskConical size={14} /> Demo</button>
                    <button disabled={!!running} className={mode === "live" ? "active" : ""} onClick={() => setMode("live")}><Search size={14} /> Live</button>
                  </div>
                </div>
                {mode === "demo" && (
                  <div className="demo-bar">
                    <FlaskConical size={16} />
                    <span>Hasil demo sintetis · untuk memeriksa alur</span>
                    <div>
                      <button disabled={busy} onClick={() => void loadDemo("fact_check")}>Contoh cek fakta</button>
                      <button disabled={busy} onClick={() => void loadDemo("news")}>Contoh berita</button>
                    </div>
                  </div>
                )}
                <label className="field-label" htmlFor="caption">TEKS KLAIM <span>maks. 10.000 karakter</span></label>
                <textarea id="caption" maxLength={10000} value={caption} disabled={!!running}
                  onChange={(e) => setCaption(e.target.value)}
                  placeholder="Contoh: Presiden menetapkan 30 September sebagai hari libur nasional" />
                <div className="field-row">
                  <label>Bahasa
                    <select value={language} disabled={!!running} onChange={(e) => setLanguage(e.target.value)}>
                      <option value="id">Bahasa Indonesia</option><option value="en">English</option>
                    </select>
                  </label>
                  <label>Batas waktu informasi (opsional)
                    <input type="date" value={asOf} disabled={!!running} onChange={(e) => setAsOf(e.target.value)} />
                  </label>
                  <label>Gambar (opsional)
                    <input ref={fileInput} type="file" accept="image/png,image/jpeg,image/webp" multiple className="sr-only"
                      aria-label="Unggah gambar" onChange={(e) => { void upload(e.target.files); e.target.value = ""; }} />
                    <button className="button secondary" disabled={busy || !!running} onClick={() => fileInput.current?.click()}>
                      <Upload size={14} /> Pilih gambar
                    </button>
                  </label>
                </div>
                {!!media.length && (
                  <div className="thumb-strip">
                    {media.map((m) => (
                      <div className="thumb-item" key={m.asset_id}>
                        <img src={`/api/v1/media/${m.asset_id}/thumbnail`} alt={`Gambar ${m.asset_id}`} />
                        <button className="thumb-remove" aria-label={`Hapus gambar ${m.asset_id}`} disabled={!!running}
                          onClick={() => setMedia(media.filter((x) => x.asset_id !== m.asset_id))}><X size={13} /></button>
                        <span>{m.width} × {m.height} px</span>
                      </div>
                    ))}
                  </div>
                )}
                <div className="input-footer">
                  <span><ShieldCheck size={15} /> {mode === "live" ? "Korpus lokal nyata + penyedia aktif · kutipan selalu persis" : "Fixture berlabel jelas, tanpa klaim akurasi"}</span>
                  <button className="button primary" onClick={() => void submit()} disabled={busy || !!running || !caption.trim()}>
                    {busy || running ? <LoaderCircle className="spin" size={17} /> : <ScanLine size={17} />}
                    {running ? "Mencari…" : "Jalankan pencarian"} <ArrowRight size={16} />
                  </button>
                </div>
              </section>

              {running && (
                <div className="progress-card" role="status">
                  <LoaderCircle className="spin" size={22} />
                  <div><strong>{running.progress}</strong><p>Pekerjaan tersimpan; halaman boleh dimuat ulang.</p></div>
                </div>
              )}

              {bundle?.retrieval && (
                <section className="card result-card">
                  <div className="card-top">
                    <div>
                      <span className="section-number">HASIL</span>
                      <h2>{evidence.length} bukti · {signals.length} sinyal deteksi AI</h2>
                    </div>
                    <div className="export-buttons">
                      {["json", "csv", "zip"].map((f) => (
                        <button key={f} onClick={() => window.open(`/api/v1/cases/${bundle.case_id}/export?format=${f}`, "_blank")}>
                          <Download size={14} /> {f.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="provider-strip">
                    {bundle.retrieval.provider_status.map((p) => (
                      <span key={p.provider + p.capability} className={`tiny-tag ${p.status}`}>{p.provider}: {statusLabel(p.status)}</span>
                    ))}
                  </div>
                  <div className="tab-row" role="tablist">
                    {([["evidence", `Bukti (${evidence.length})`], ["image", `Asal Gambar (${imageMatches.length})`], ["ai", `Deteksi AI (${signals.length})`], ["trace", "Jejak Pencarian"]] as const).map(([key, label]) => (
                      <button key={key} role="tab" aria-selected={tab === key} className={tab === key ? "active" : ""} onClick={() => setTab(key)}>{label}</button>
                    ))}
                  </div>

                  {tab === "evidence" && (
                    <>
                      <div className="field-row">
                        <label>Filter jenis
                          <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}>
                            <option value="all">Semua jenis</option>
                            {Object.entries(KIND_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                          </select>
                        </label>
                        <label>Urutkan
                          <select value={sort} onChange={(e) => setSort(e.target.value as "relevance" | "date")}>
                            <option value="relevance">Paling relevan</option>
                            <option value="date">Terbaru</option>
                          </select>
                        </label>
                      </div>
                      <div className="evidence-list">
                        {shown.map((e) => (
                          <article className="evidence-item" key={e.evidence_id}>
                            <header>
                              <span className="tiny-tag">{KIND_LABEL[e.source.kind] || e.source.kind}</span>
                              <strong>{e.source.title}</strong>
                            </header>
                            <p className="evidence-meta">
                              {e.source.publisher || "Penerbit tidak diketahui"} · terbit {fmtDate(e.source.published_at)} · diambil {fmtDate(e.source.retrieved_at)}
                            </p>
                            {e.source.url && (
                              <a href={e.source.url} target="_blank" rel="noreferrer noopener">Buka sumber asli ↗</a>
                            )}
                            <blockquote>
                              “{e.content.excerpt || "(kutipan tidak tersedia)"}”
                              <small>
                                {e.content.status === "snippet_only" ? "Kutipan cuplikan saja — halaman lengkap belum dibaca" : e.content.status === "full" ? "Teks lengkap dibaca" : "Konten tidak tersedia"}
                              </small>
                            </blockquote>
                            <dl className="evidence-scores">
                              <div><dt>Relevansi</dt><dd>{e.relevance_score == null ? "Tidak tersedia" : e.relevance_score.toFixed(3)}</dd></div>
                              <div><dt>Kelayakan waktu</dt><dd>{e.provenance.temporal_eligible === null ? "Tidak diketahui" : e.provenance.temporal_eligible ? "Layak" : "Terbit setelah batas waktu"}</dd></div>
                              <div><dt>Kelompok independen</dt><dd title={e.provenance.independence_group_id}>{e.provenance.independence_group_id.slice(0, 12)}…</dd></div>
                              <div><dt>Kecocokan poin klaim</dt><dd>{e.atom_ids.length || "—"}</dd></div>
                            </dl>
                            <details>
                              <summary>Rincian peringkat</summary>
                              <ul>
                                {Object.entries(e.score_breakdown).map(([k, v]) => <li key={k}>{k}: {Number(v).toFixed(3)}</li>)}
                                <li>metode penemuan: {e.provenance.discovery_method}</li>
                              </ul>
                            </details>
                          </article>
                        ))}
                        {!shown.length && <p className="empty-note">Tidak ada bukti yang cocok dengan filter ini.</p>}
                      </div>
                    </>
                  )}

                  {tab === "image" && (
                    <div className="evidence-list">
                      {imageMatches.map((m, i) => (
                        <article className="evidence-item" key={i}>
                          <header>
                            <span className="tiny-tag">{m.match_type === "exact" ? "Sama persis" : m.match_type === "near_duplicate" ? "Hampir sama" : "Mirip secara semantik"}</span>
                            <strong>{m.evidence.source.title}</strong>
                          </header>
                          <p className="evidence-meta">skor kemiripan {m.score?.toFixed(3) ?? "tidak tersedia"} · metode dHash korpus lokal</p>
                          {m.matched_url && <a href={m.matched_url} target="_blank" rel="noreferrer noopener">Buka halaman sumber ↗</a>}
                          <p className="evidence-meta">Pencarian gambar ini terbatas pada korpus lokal, bukan seluruh web.</p>
                        </article>
                      ))}
                      {!imageMatches.length && <p className="empty-note">Tidak ada kecocokan gambar dari korpus lokal. Jalankan pencarian dengan gambar untuk mencoba.</p>}
                    </div>
                  )}

                  {tab === "ai" && (
                    <div className="ai-panels">
                      {([["Gambar", imageSignals], ["Teks klaim", textSignals]] as const).map(([label, list]) => (
                        <div className="ai-panel" key={label}>
                          <h3>Deteksi AI · {label}</h3>
                          {list.map((s) => (
                            <div className="signal-item" key={s.signal_id}>
                              <header>
                                <strong>{ASSESSMENT_LABEL[s.assessment] || s.assessment}</strong>
                                <span className={`tiny-tag ${s.status}`}>{statusLabel(s.status)}</span>
                              </header>
                              <dl>
                                <div><dt>Penyedia</dt><dd>{s.provider}{s.model_version ? ` · ${s.model_version}` : ""}</dd></div>
                                <div><dt>Skor asli</dt><dd>{s.raw_score == null ? "—" : `${s.raw_score.toFixed(3)} (skala ${s.raw_scale?.min}–${s.raw_scale?.max}${s.raw_scale?.higher_means_ai ? ", makin tinggi makin AI" : ", makin rendah makin AI"})`}</dd></div>
                                <div><dt>Indikasi AI</dt><dd>{s.ai_generated_score == null ? "Tidak dinilai" : s.ai_generated_score.toFixed(3)}</dd></div>
                                <div><dt>Kalibrasi</dt><dd>{s.calibration_status === "provider_claimed" ? "Klaim penyedia, belum divalidasi lokal" : s.calibration_status}</dd></div>
                              </dl>
                              {s.error_code && <p className="evidence-meta">kode: {s.error_code}</p>}
                              <ul>
                                {s.limitations.map((l, i) => <li key={i}>{l}</li>)}
                              </ul>
                              <p className="evidence-meta">Indikasi konten AI bukan bukti klaim palsu; konten kamera/manusia juga bukan bukti klaim benar.</p>
                            </div>
                          ))}
                          {!list.length && <p className="empty-note">Tidak ada sinyal untuk {label.toLowerCase()}.</p>}
                        </div>
                      ))}
                    </div>
                  )}

                  {tab === "trace" && (
                    <div className="trace-list">
                      <h3>Kueri yang dijalankan</h3>
                      {bundle.retrieval.query_log.map((q) => (
                        <div className="trace-item" key={q.query_id}>
                          <code>{q.query}</code>
                          <span>{q.provider} · {q.result_count} hasil · {Math.round(q.duration_ms)} ms</span>
                        </div>
                      ))}
                      <h3>Status penyedia</h3>
                      {bundle.retrieval.provider_status.map((p, i) => (
                        <div className="trace-item" key={i}>
                          <code>{p.provider}</code>
                          <span>{statusLabel(p.status)}{p.message ? ` · ${p.message}` : ""}</span>
                        </div>
                      ))}
                      <h3>Keterbatasan</h3>
                      <ul>
                        {bundle.retrieval.run.warnings.map((w, i) => <li key={i}>{w.message}</li>)}
                        <li>Korpus demo berlabel sintetis; hasil live memerlukan kunci penyedia.</li>
                      </ul>
                    </div>
                  )}
                </section>
              )}
              <div className="import-footer">
                <span>Punya bundle dari layanan AURORA lain?</span>
                <input ref={importInput} type="file" accept=".json,.zip" className="sr-only" aria-label="Impor berkas hasil"
                  onChange={(e) => { void importBundle(e.target.files?.[0]); e.target.value = ""; }} />
                <button className="text-button" onClick={() => importInput.current?.click()}><Upload size={14} /> Impor berkas hasil (JSON/ZIP)</button>
              </div>
            </>
          )}

          {page === "history" && (
            <section className="card history-page">
              <div className="card-top"><h2>Semua kasus <span className="tiny-tag">{history.length}</span></h2></div>
              {history.map((h) => (
                <button className="history-row" key={h.case_id} onClick={() => void loadCase(h.case_id)}>
                  <span className="history-file"><FileJson size={22} /></span>
                  <span><strong>{h.caption}</strong><small>{new Date(h.updated_at * 1000).toLocaleString("id-ID")} · Revisi berapa pun tetap satu kasus</small></span>
                  <span className="tiny-tag">{h.mode}</span>
                  <ChevronRight size={18} />
                </button>
              ))}
              {!history.length && <div className="empty-history"><Clock3 size={36} /><h3>Belum ada kasus tersimpan</h3><p>Mulai dengan menjalankan pencarian.</p></div>}
            </section>
          )}

          {page === "settings" && (
            <section className="card settings-groups">
              <h3>Status penyedia</h3>
              <div className="capability-grid">
                {capabilities.map((c) => (
                  <div className="card capability-card" key={c.provider}>
                    <div><span className="tiny-tag">{c.mode}</span><span className={`cap-status ${c.status}`}>{statusLabel(c.status)}</span></div>
                    <h3>{c.provider}</h3><p>{c.message}</p><small>{c.capability}</small>
                  </div>
                ))}
              </div>
              <p className="evidence-meta"><AlertTriangle size={14} /> Kunci API hanya diatur lewat variabel lingkungan server (TAVILY_API_KEY, GPTZERO_API_KEY, HIVE_API_KEY) dan tidak pernah tampil di UI.</p>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
