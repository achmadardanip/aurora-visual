import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronRight,
  CircleHelp,
  Clock3,
  FileJson,
  Fingerprint,
  FlaskConical,
  Focus,
  History,
  ImagePlus,
  Layers3,
  LoaderCircle,
  Pencil,
  Plus,
  ScanLine,
  ScanText,
  Settings2,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
import type {
  Atom,
  AuroraBundle,
  MediaRef,
  VisualAssessment,
} from "../../contracts/aurora";
import { api, base, download, headers } from "./api";

type Page = "analysis" | "history" | "methods";
type Capability = {
  provider: string;
  mode: string;
  capability: string;
  status: string;
  message: string | null;
};
type Screening = {
  decision: {
    label: "likely_ai_generated" | "inconclusive" | "no_strong_ai_signal";
    rationale: string;
    does_not_affect_visual_assessment: boolean;
    does_not_decide_claim_truth: boolean;
  };
  metadata: {
    status: string;
    field_names: string[];
    info_field_names: string[];
    camera_metadata_present: boolean;
    capture_time_present: boolean;
    software_classification: string;
  };
  provenance: {
    c2pa: { status: string; verification: string; message: string };
    watermark: { status: string; message: string };
  };
  detectors: {
    task: string;
    provider: string;
    status: string;
    assessment?: string;
    score?: number | null;
    message: string;
  }[];
  limitations: string[];
};
type HiveStatus = {
  capability: string;
  status: string;
  error_code?: string;
};
type HiveExtension = {
  mode: string;
  provider_status?: HiveStatus[];
  stage1?: {
    provider: string;
    status: string;
    representation?: string;
    coordinate_space?: string;
  } | null;
  stage2?: {
    provider: string;
    status: string;
    atom_count?: number;
    error_code?: string;
  } | null;
  stage3?: {
    provider?: string;
    representation?: string;
    status?: string;
    models?: { model: string; capabilities: string[] }[];
    vlm?: {
      provider: string;
      status: string;
      observations?: unknown[];
      interpretation?: string;
    };
  } | null;
  translation_shadow?: {
    status: string;
    canonical_caption_unchanged: boolean;
  };
};
type CaseItem = {
  case_id: string;
  caption: string;
  mode: string;
  claim_revision: number;
  updated_at: number;
  status: string;
};
type Job = {
  job_id: string;
  status: string;
  progress: string;
  error: { message: string } | null;
  result: AuroraBundle | null;
};
const labels = {
  Supported: "Didukung visual",
  Contradicted: "Bertentangan",
  Unobservable: "Tidak teramati",
};
const screeningLabels = {
  likely_ai_generated: "Kemungkinan media generatif",
  inconclusive: "Belum konklusif",
  no_strong_ai_signal: "Tidak ada sinyal AI kuat",
};
const roleNames: Record<string, string> = {
  actor: "Pelaku / identitas",
  action: "Aksi",
  object: "Objek",
  attribute: "Atribut",
  location: "Lokasi",
  time: "Waktu",
  quantity: "Jumlah",
  relation: "Relasi",
  cause: "Sebab",
};
const detectorStatus: Record<string, string> = {
  ok: "Teramati",
  inconclusive: "Belum konklusif",
  unsupported: "Tidak didukung project",
  unsupported_output: "Output tidak didukung",
  unavailable: "Belum dikonfigurasi",
  unconfigured: "Belum dikonfigurasi",
  unconfigured_or_unsupported: "Belum tersedia",
  failed: "Provider gagal",
  provider_failed: "Provider gagal",
  http_error: "Provider gagal",
  network_error: "Jaringan gagal",
  timeout: "Provider timeout",
  rate_limited: "Rate limited",
  malformed_response: "Respons tidak valid",
  observation_invalid: "Observasi tidak valid",
  failed_fallback_rules: "Gagal · aturan lokal dipakai",
  not_applicable: "Tidak berlaku",
};
const statusLabel = (status?: string) =>
  status
    ? detectorStatus[status] || status.replaceAll("_", " ")
    : "Belum tersedia";
const score = (v: number | null | undefined) =>
  v == null ? "Tidak tersedia" : v.toFixed(3);

function MediaImage({
  media,
  assessments,
  selected,
}: {
  media: MediaRef;
  assessments: VisualAssessment[];
  selected: string | null;
}) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    let uri = "",
      active = true;
    fetch(`${base}/api/v1/media/${media.asset_id}?preview=true`, {
      headers: headers(),
    })
      .then(async (response) => {
        if (!response.ok) throw new Error("Media tidak tersedia");
        uri = URL.createObjectURL(await response.blob());
        if (active) setUrl(uri);
      })
      .catch(() => {
        if (active) setUrl("");
      });
    return () => {
      active = false;
      if (uri) URL.revokeObjectURL(uri);
    };
  }, [media.asset_id]);
  const assessment = assessments.find((a) => a.atom_id === selected);
  const regions = [
    ...(assessment?.supporting_regions || []).map((r) => ({
      ...r,
      kind: "support",
    })),
    ...(assessment?.contradicting_regions || []).map((r) => ({
      ...r,
      kind: "contra",
    })),
  ];
  return (
    <div className="image-frame">
      {url ? (
        <div className="image-relative">
          <img src={url} alt="Gambar masukan yang sedang diperiksa" />
          {regions.map((r) => (
            <span
              key={r.region_id}
              className={`region ${r.kind}`}
              title={r.description}
              style={{
                left: `${r.bbox[0] * 100}%`,
                top: `${r.bbox[1] * 100}%`,
                width: `${(r.bbox[2] - r.bbox[0]) * 100}%`,
                height: `${(r.bbox[3] - r.bbox[1]) * 100}%`,
              }}
            />
          ))}
        </div>
      ) : (
        <p>Media belum tersedia. Unggah gambar dengan hash yang sesuai.</p>
      )}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("analysis");
  const [mode, setMode] = useState<"demo" | "live">("demo");
  const [caption, setCaption] = useState("");
  const [language, setLanguage] = useState("id");
  const [media, setMedia] = useState<MediaRef | null>(null);
  const [bundle, setBundle] = useState<AuroraBundle | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [history, setHistory] = useState<CaseItem[]>([]);
  const [capabilities, setCapabilities] = useState<Capability[]>([]);
  const [ready, setReady] = useState("Menghubungkan");
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [backbone, setBackbone] = useState("local-color-v1");
  const [alignment, setAlignment] = useState("uot");
  const [provider, setProvider] = useState<"local" | "hive">("local");
  const [editing, setEditing] = useState<Atom[] | null>(null);
  const [reason, setReason] = useState("");
  const [snapshots, setSnapshots] = useState<
    { kind: string; reason: string | null; bundle: AuroraBundle }[]
  >([]);
  const [showSnapshots, setShowSnapshots] = useState(false);
  const lastRequest = useRef<{ payload: AuroraBundle; key: string } | null>(
    null,
  );
  const modalRef = useRef<HTMLElement>(null);
  const inputFile = useRef<HTMLInputElement>(null);
  const importFile = useRef<HTMLInputElement>(null);
  const running = job?.status === "queued" || job?.status === "running";
  const analysis = bundle?.analysis;
  const visual = analysis?.visual_assessments || [];
  const atom = analysis?.atomic_claims.find((a) => a.atom_id === selected);
  const assessment = visual.find((a) => a.atom_id === selected);
  const extension = bundle?.extensions.aurora_visual as
    | {
        method?: {
          backbone: string;
          alignment: string;
          mode: string;
          region_method: string;
          calibration: string;
        };
        timing?: { total_ms: number };
        screening?: Screening;
        hive?: HiveExtension | null;
        correction?: { reason: string };
      }
    | undefined;
  const screening = extension?.screening;
  const hive = extension?.hive;
  const hiveV3Ready = capabilities.some(
    (capability) =>
      capability.provider === "hive-v3-vlm" && capability.status === "ok",
  );
  const activeProvider =
    mode === "live" && provider === "hive" ? "Hive eksternal" : "Lokal";
  const detectorSummary = screening?.detectors
    .map((detector) => statusLabel(detector.status))
    .filter((value, index, values) => values.indexOf(value) === index)
    .join(" · ");

  const editorOpen = editing !== null;
  useEffect(() => {
    if (!editorOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    const panel = modalRef.current;
    const focusables = () =>
      [
        ...(panel?.querySelectorAll<HTMLElement>(
          "button:not(:disabled),input,textarea,select,summary",
        ) || []),
      ].filter((e) => e.offsetParent !== null);
    focusables()[0]?.focus();
    const trap = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const elements = focusables(),
        first = elements[0],
        last = elements[elements.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      }
      if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", trap);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", trap);
      document.body.style.overflow = overflow;
      previous?.focus();
    };
  }, [editorOpen]);

  async function refresh() {
    try {
      const state = await api<{ status: string; capabilities: Capability[] }>(
        "/ready",
      );
      setCapabilities(state.capabilities);
      setReady(
        state.status === "ready"
          ? "Siap"
          : "Lokal siap · fitur opsional terbatas",
      );
      setHistory(await api<CaseItem[]>("/api/v1/cases"));
    } catch (e) {
      setReady("Layanan belum siap");
      setError((e as Error).message);
    }
  }
  useEffect(() => {
    void refresh();
  }, []);
  useEffect(() => {
    const saved = sessionStorage.getItem("aurora-job");
    if (saved) {
      setJob({
        job_id: saved,
        status: "queued",
        progress: "Memulihkan status pekerjaan",
        result: null,
        error: null,
      });
    }
    const id = location.hash.startsWith("#case/") ? location.hash.slice(6) : "";
    if (id) void loadCase(id);
  }, []);
  useEffect(() => {
    if (!running || !job) return;
    let active = true;
    const poll = async () => {
      try {
        const next = await api<Job>(`/api/v1/jobs/${job.job_id}`);
        if (!active) return;
        setJob(next);
        if (!["queued", "running"].includes(next.status)) {
          sessionStorage.removeItem("aurora-job");
          if (next.result) {
            display(next.result);
            setNotice(
              next.status === "partial"
                ? "Analisis parsial. Periksa peringatan komponen."
                : "Analisis selesai. Pilih atom untuk menelusuri bukti.",
            );
          }
          if (next.error) setError(next.error.message);
          void refresh();
        }
      } catch (e) {
        if (active) {
          setError((e as Error).message);
        }
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), 1000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [job?.job_id, running]);

  function display(data: AuroraBundle) {
    setBundle(data);
    setCaption(data.input.claim_text);
    setLanguage(data.input.language);
    setMedia(data.input.image);
    setMode(data.mode);
    const savedOptions = (
      data.extensions.aurora_visual as
        { options?: { provider?: "local" | "hive" } } | undefined
    )?.options;
    setProvider(
      data.mode === "demo" ? "local" : savedOptions?.provider || "local",
    );
    setSelected(data.analysis?.atomic_claims[0]?.atom_id || null);
    setEditing(null);
    setShowSnapshots(false);
    setPage("analysis");
    location.hash = `case/${data.case_id}`;
  }
  async function guard(action: () => Promise<void>) {
    setError("");
    setNotice("");
    setBusy(true);
    try {
      await action();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function loadCase(id: string) {
    await guard(async () =>
      display(await api<AuroraBundle>(`/api/v1/cases/${id}`)),
    );
  }
  function reset() {
    if (running) return;
    setBundle(null);
    setMedia(null);
    setCaption("");
    setJob(null);
    setError("");
    setNotice("");
    setEditing(null);
    setPage("analysis");
    location.hash = "";
    lastRequest.current = null;
  }
  async function upload(file?: File) {
    if (!file) return;
    await guard(async () => {
      if (file.size > 10 * 1024 ** 2)
        throw new Error("Ukuran maksimum gambar adalah 10 MB.");
      const form = new FormData();
      form.append("image", file);
      setMedia(
        await api<MediaRef>("/api/v1/media", { method: "POST", body: form }),
      );
    });
  }
  async function demo(fixture: string) {
    await guard(async () => {
      display(
        await api<AuroraBundle>(`/api/v1/demo/${fixture}`, { method: "POST" }),
      );
      setJob(null);
    });
  }
  async function submit(retry = false) {
    await guard(async () => {
      if (!media || !caption.trim())
        throw new Error("Unggah gambar dan isi caption terlebih dahulu.");
      let payload: AuroraBundle;
      if (bundle) {
        const changed =
          caption !== bundle.input.claim_text ||
          media.sha256 !== bundle.input.image?.sha256;
        payload = {
          ...bundle,
          claim_revision: bundle.claim_revision + (changed ? 1 : 0),
          input: {
            ...bundle.input,
            claim_text: caption,
            image: media,
            language,
          },
          analysis: changed ? null : bundle.analysis,
          retrieval: null,
          decision: null,
        };
      } else
        payload = {
          schema_version: "1.0.0",
          case_id: crypto.randomUUID(),
          claim_revision: 1,
          mode,
          created_at: new Date().toISOString(),
          input: { claim_text: caption, language, image: media, as_of: null },
          analysis: null,
          retrieval: null,
          decision: null,
          warnings: [],
          extensions: {},
        };
      payload.extensions = {
        ...payload.extensions,
        aurora_visual: {
          ...((payload.extensions.aurora_visual as object) || {}),
          options: {
            backbone,
            alignment,
            parser: provider === "hive" ? "hive-vlm" : "rules",
            head: "heuristic",
            top_k: 16,
            provider: mode === "demo" ? "local" : provider,
            translation_shadow: false,
          },
        },
      };
      const request =
        retry && lastRequest.current
          ? lastRequest.current
          : { payload, key: crypto.randomUUID() };
      lastRequest.current = request;
      const response = await api<{ job_id: string }>("/api/v1/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": request.key,
        },
        body: JSON.stringify(request.payload),
      });
      sessionStorage.setItem("aurora-job", response.job_id);
      setJob({
        job_id: response.job_id,
        status: "queued",
        progress: "Menunggu worker",
        result: null,
        error: null,
      });
    });
  }
  async function importBundle(file?: File) {
    if (!file) return;
    await guard(async () => {
      const form = new FormData();
      form.append("file", file);
      const result = await api<{
        bundle: AuroraBundle;
        asset_available: boolean;
      }>("/api/v1/import", { method: "POST", body: form });
      display(result.bundle);
      setNotice(
        result.asset_available
          ? "Bundle dan media berhasil diimpor."
          : "Metadata berhasil diimpor. Media belum tersedia untuk analisis ulang.",
      );
      await refresh();
    });
  }
  async function saveAtoms() {
    if (!bundle?.analysis || !editing) return;
    await guard(async () => {
      const result = await api<AuroraBundle>(
        `/api/v1/cases/${bundle.case_id}/atoms`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            atomic_claims: editing,
            expected_atom_set_id: bundle.analysis!.atom_set_id,
            reason,
          }),
        },
      );
      display(result);
      setReason("");
      setNotice(
        "Koreksi tersimpan sebagai atom set baru. Jalankan analisis ulang.",
      );
    });
  }
  const exportResult = (format: string) =>
    guard(() =>
      download(
        `/api/v1/cases/${bundle!.case_id}/export?format=${format}${selected ? `&atom_id=${selected}` : ""}`,
        `aurora-${bundle!.case_id}.${format === "overlay" ? "png" : format}`,
      ),
    );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          href="#"
          className="brand"
          aria-label="AURORA, halaman analisis"
          onClick={(e) => {
            e.preventDefault();
            setPage("analysis");
          }}
        >
          <span className="brand-mark">
            <ScanLine size={25} />
          </span>
          <span>
            AURORA<small>VISUAL INTELLIGENCE</small>
          </span>
        </a>
        <div className="workspace-label">RUANG KERJA PENELITIAN</div>
        <nav aria-label="Navigasi utama">
          {(
            [
              { id: "analysis", label: "Analisis visual", Icon: Focus },
              { id: "history", label: "Riwayat kasus", Icon: History },
              { id: "methods", label: "Metode & kemampuan", Icon: Settings2 },
            ] as const
          ).map(({ id, label, Icon }) => (
            <button
              className={page === id ? "nav-item active" : "nav-item"}
              key={id}
              onClick={() => setPage(id)}
            >
              <Icon size={18} />
              {label}
              {page === id && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-divider" />
        <div className="workspace-label recent-label">
          KASUS TERAKHIR{" "}
          <span>{history.length.toString().padStart(2, "0")}</span>
        </div>
        <div className="recent-list">
          {history.slice(0, 4).map((item) => (
            <button
              key={item.case_id}
              onClick={() => void loadCase(item.case_id)}
            >
              <span className="recent-icon">
                <FileJson size={15} />
              </span>
              <span>
                {item.caption}
                <small>
                  {item.mode === "demo" ? "DEMO" : "LIVE"} · Revisi{" "}
                  {item.claim_revision}
                </small>
              </span>
            </button>
          ))}
          {!history.length && (
            <p>Kasus yang dianalisis akan tersimpan di sini.</p>
          )}
        </div>
        <div className="sidebar-note">
          <ShieldCheck size={21} />
          <strong>Bukti yang bisa ditelusuri.</strong>
          <p>
            Setiap klaim diperiksa terpisah. Hasil visual bukan verdict faktual
            akhir.
          </p>
          <span>
            MODUL 01 <i /> KONTRAK 1.0.0
          </span>
        </div>
        <div className="sidebar-footer">
          <span className="online-dot" /> {activeProvider} <span>v1.0</span>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            AURORA <ChevronRight size={14} />{" "}
            <strong>
              {page === "analysis"
                ? "Analisis visual"
                : page === "history"
                  ? "Riwayat kasus"
                  : "Metode & kemampuan"}
            </strong>
          </div>
          <span
            className={`local-pill ${provider === "hive" && mode === "live" ? "external" : ""}`}
          >
            <span className="online-dot" /> {activeProvider}
          </span>
        </header>
        <div className="main-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                <span /> IMAGE–CAPTION CONSISTENCY
              </div>
              <h1>
                {page === "analysis"
                  ? "Periksa klaim, satu per satu."
                  : page === "history"
                    ? "Jejak setiap pemeriksaan."
                    : "Metode yang transparan."}
              </h1>
              <p>
                {page === "analysis"
                  ? "Hubungkan klaim atomik dengan bukti yang benar-benar terlihat."
                  : page === "history"
                    ? "Kasus, revisi, dan hasil analisis tersimpan di workspace Anda."
                    : "Ketahui apa yang dijalankan, apa yang tersedia, dan batas hasilnya."}
              </p>
            </div>
            <button
              className="button secondary new-case"
              onClick={reset}
              disabled={!!running}
            >
              <Plus size={16} /> Kasus baru
            </button>
          </div>
          {error && (
            <div className="banner error" role="alert">
              <CircleHelp size={18} />
              <span>{error}</span>
              <button aria-label="Tutup error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {notice && (
            <div className="banner notice" role="status">
              <Check size={18} />
              <span>{notice}</span>
            </div>
          )}
          {page === "analysis" && (
            <>
              <div
                className="workflow-strip"
                aria-label="Tahapan analisis AURORA"
              >
                <span className={media ? "current" : ""}>
                  <span>0</span> Masukan
                </span>
                <i />
                <span className={extension?.screening ? "current" : ""}>
                  <span>1</span> Asal media
                </span>
                <i />
                <span className={analysis ? "current" : ""}>
                  <span>2</span> Urai klaim
                </span>
                <i />
                <span className={visual.length ? "current" : ""}>
                  <span>3</span> Analisis multimodal
                </span>
                <div>
                  <Layers3 size={14} /> OCR · region · UOT
                </div>
              </div>
              <section className="card input-card">
                <div className="card-top">
                  <div>
                    <span className="section-number">TAHAP 0</span>
                    <h2>Masukan analisis</h2>
                  </div>
                  <div className="mode-switch" aria-label="Mode analisis">
                    <button
                      disabled={!!bundle || !!running}
                      className={mode === "demo" ? "active" : ""}
                      onClick={() => {
                        setMode("demo");
                        setProvider("local");
                      }}
                    >
                      <FlaskConical size={14} /> Demo
                    </button>
                    <button
                      disabled={!!bundle || !!running}
                      className={mode === "live" ? "active" : ""}
                      onClick={() => setMode("live")}
                    >
                      <Activity size={14} /> Live
                    </button>
                  </div>
                </div>
                {mode === "demo" && (
                  <div className="demo-bar">
                    <FlaskConical size={16} />
                    <span>Demo sintetis · untuk memeriksa alur aplikasi</span>
                    <div>
                      <button
                        disabled={busy || !!running}
                        onClick={() => void demo("supported")}
                      >
                        Dukungan
                      </button>
                      <button
                        disabled={busy || !!running}
                        onClick={() => void demo("contradicted")}
                      >
                        Bantahan
                      </button>
                      <button
                        disabled={busy || !!running}
                        onClick={() => void demo("unobservable")}
                      >
                        Tak teramati
                      </button>
                    </div>
                  </div>
                )}
                <div className="input-grid">
                  <div>
                    <label className="field-label">
                      GAMBAR SUMBER <span>PNG, JPG, WebP · maks. 10 MB</span>
                    </label>
                    <input
                      ref={inputFile}
                      type="file"
                      accept="image/png,image/jpeg,image/webp"
                      className="sr-only"
                      aria-label="Unggah gambar"
                      onChange={(e) => void upload(e.target.files?.[0])}
                    />
                    {media ? (
                      <div className="input-image">
                        <MediaImage
                          media={media}
                          assessments={[]}
                          selected={null}
                        />
                        <button
                          className="image-replace"
                          disabled={busy || !!running}
                          onClick={() => inputFile.current?.click()}
                        >
                          <Upload size={14} /> Ganti gambar
                        </button>
                        <span>
                          {media.width} × {media.height} px
                        </span>
                      </div>
                    ) : (
                      <button
                        className="upload-zone"
                        disabled={busy || !!running}
                        onClick={() => inputFile.current?.click()}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={(e) => {
                          e.preventDefault();
                          void upload(e.dataTransfer.files[0]);
                        }}
                      >
                        <span className="upload-icon">
                          <ImagePlus size={30} />
                        </span>
                        <strong>Pilih atau letakkan gambar</strong>
                        <span>
                          {mode === "live" && provider === "hive"
                            ? "Unggahan disimpan di AURORA; analisis Hive hanya setelah persetujuan"
                            : "Gambar akan diproses di workspace lokal"}
                        </span>
                        <em>
                          <Plus size={14} /> Pilih berkas
                        </em>
                      </button>
                    )}
                  </div>
                  <div className="caption-field">
                    <label className="field-label" htmlFor="caption">
                      CAPTION ASLI{" "}
                      <span>{[...caption].length}/10.000 karakter</span>
                    </label>
                    <textarea
                      id="caption"
                      maxLength={10000}
                      value={caption}
                      disabled={!!running}
                      onChange={(e) => setCaption(e.target.value)}
                      placeholder="Contoh: Mahasiswa melakukan aksi di Monas pada September 2026"
                    />
                    <div className="caption-note">
                      <CircleHelp size={14} />
                      <span>
                        Caption asli disimpan. Waktu, lokasi, dan identitas
                        memerlukan bukti khusus.
                      </span>
                    </div>
                    <div className="field-row">
                      <label>
                        Bahasa
                        <select
                          value={language}
                          onChange={(e) => setLanguage(e.target.value)}
                          disabled={!!running}
                        >
                          <option value="id">Bahasa Indonesia</option>
                          <option value="en">English</option>
                        </select>
                      </label>
                      <label>
                        Model visual
                        <select
                          value={backbone}
                          onChange={(e) => setBackbone(e.target.value)}
                          disabled={!!running}
                        >
                          <option value="local-color-v1">
                            Lokal · warna bidang
                          </option>
                          <option
                            value="openclip"
                            disabled={
                              !capabilities.some(
                                (c) =>
                                  c.provider === "openclip" &&
                                  c.status === "ok",
                              )
                            }
                          >
                            OpenCLIP{" "}
                            {capabilities.some(
                              (c) =>
                                c.provider === "openclip" && c.status === "ok",
                            )
                              ? ""
                              : "· belum tersedia"}
                          </option>
                        </select>
                      </label>
                    </div>
                    {mode === "live" && (
                      <div className="provider-choice">
                        <span>Pemrosesan</span>
                        <label>
                          <input
                            type="radio"
                            name="provider"
                            value="local"
                            checked={provider === "local"}
                            onChange={() => setProvider("local")}
                            disabled={!!running}
                          />
                          Lokal
                        </label>
                        <label>
                          <input
                            type="radio"
                            name="provider"
                            value="hive"
                            checked={provider === "hive"}
                            onChange={() => setProvider("hive")}
                            disabled={!!running || !hiveV3Ready}
                          />
                          Hive eksternal
                        </label>
                      </div>
                    )}
                    {mode === "live" && !hiveV3Ready && (
                      <p className="provider-unavailable">
                        Hive belum dapat dipilih: server belum memiliki
                        konfigurasi V3 VLM lengkap.
                      </p>
                    )}
                    {mode === "live" && provider === "hive" && (
                      <div className="egress-disclosure" role="note">
                        <ShieldCheck size={16} />
                        <span>
                          Dengan menjalankan analisis, byte gambar asli dikirim
                          untuk Tahap 1; preview ternormalisasi dan caption
                          dikirim untuk Tahap 2–3 ke Hive. Kredensial tetap di
                          server. Kebijakan retensi provider berlaku.
                        </span>
                      </div>
                    )}
                  </div>
                </div>
                <div className="input-footer">
                  <span>
                    <ShieldCheck size={15} />{" "}
                    {mode === "live"
                      ? provider === "hive"
                        ? "Pemrosesan eksternal dipilih · hasil probabilistik dan jalur lokal tetap diaudit"
                        : "Komputasi lokal nyata · hasil heuristik konservatif"
                      : "Fixture berlabel jelas, tanpa klaim akurasi"}
                  </span>
                  <button
                    className="button primary"
                    onClick={() => void submit()}
                    disabled={busy || !!running || !media || !caption.trim()}
                  >
                    {busy || running ? (
                      <LoaderCircle className="spin" size={17} />
                    ) : (
                      <ScanLine size={17} />
                    )}{" "}
                    {running
                      ? "Menganalisis…"
                      : analysis
                        ? "Analisis ulang"
                        : "Jalankan analisis"}{" "}
                    <ArrowRight size={16} />
                  </button>
                </div>
              </section>
              {running && (
                <div className="progress-card" role="status">
                  <LoaderCircle className="spin" size={22} />
                  <div>
                    <strong>{job.progress}</strong>
                    <p>Pekerjaan tersimpan; halaman boleh dimuat ulang.</p>
                  </div>
                  <button
                    onClick={() =>
                      void guard(async () => {
                        await api(`/api/v1/jobs/${job.job_id}/cancel`, {
                          method: "POST",
                        });
                      })
                    }
                  >
                    Batalkan
                  </button>
                </div>
              )}
              {job?.status === "failed" && (
                <button
                  className="button secondary"
                  onClick={() => void submit()}
                  disabled={busy}
                >
                  Coba analisis baru <ArrowRight size={15} />
                </button>
              )}
              {analysis ? (
                <section className="results-section">
                  {screening && (
                    <section
                      className="screening-panel"
                      aria-labelledby="screening-title"
                    >
                      <div className="screening-kicker">
                        <Fingerprint size={18} />
                        <span>TAHAP 1 · SCREENING ASAL MEDIA</span>
                      </div>
                      <div className="screening-summary">
                        <div>
                          <h2 id="screening-title">
                            {screeningLabels[screening.decision.label]}
                          </h2>
                          <p>{screening.decision.rationale}</p>
                        </div>
                        <span
                          className={`screening-status ${screening.decision.label}`}
                        >
                          {screening.decision.label === "likely_ai_generated"
                            ? "Perlu telaah"
                            : screening.decision.label === "no_strong_ai_signal"
                              ? "Sinyal lokal"
                              : "Tidak pasti"}
                        </span>
                      </div>
                      <div className="screening-ledger">
                        <div>
                          <span>Metadata</span>
                          <strong>
                            {screening.metadata.status === "observed"
                              ? `${screening.metadata.field_names.length} field diamati`
                              : "Tidak dapat dibaca"}
                          </strong>
                          <small>
                            {screening.metadata.camera_metadata_present
                              ? "Metadata kamera ada"
                              : "Metadata kamera tidak tersedia"}
                          </small>
                        </div>
                        <div>
                          <span>Provenance / C2PA</span>
                          <strong>{screening.provenance.c2pa.status}</strong>
                          <small>
                            {screening.provenance.c2pa.verification}
                          </small>
                        </div>
                        <div>
                          <span>Detektor AI / deepfake</span>
                          <strong>{detectorSummary || "Belum tersedia"}</strong>
                          <small>
                            {screening.detectors
                              .map((detector) =>
                                detector.task === "ai_generation_detection"
                                  ? "Generasi AI"
                                  : "Deepfake",
                              )
                              .join(" · ")}{" "}
                            · bukan verdict autentisitas
                          </small>
                        </div>
                      </div>
                      <p className="screening-boundary">
                        <CircleHelp size={15} />
                        Screening asal media tidak menentukan kebenaran caption
                        dan tidak mengubah status Didukung, Bertentangan, atau
                        Tidak teramati.
                      </p>
                      <details className="screening-details">
                        <summary>
                          Detail metadata, provenance, dan keterbatasan
                        </summary>
                        <p>
                          Perangkat lunak:{" "}
                          {screening.metadata.software_classification} · waktu
                          tangkap:{" "}
                          {screening.metadata.capture_time_present
                            ? " tersedia"
                            : " tidak tersedia"}{" "}
                          · watermark: {screening.provenance.watermark.status}
                        </p>
                        <ul>
                          {screening.limitations.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </details>
                    </section>
                  )}
                  {hive && (
                    <section
                      className="hive-provenance"
                      aria-labelledby="hive-title"
                    >
                      <div className="hive-provenance-title">
                        <ShieldCheck size={17} />
                        <div>
                          <h3 id="hive-title">Jejak pemrosesan Hive</h3>
                          <p>
                            Dukungan dikonfigurasi dan diuji dengan mock;
                            kompatibilitas live belum diverifikasi sampai
                            project key berhasil digunakan secara eksplisit.
                          </p>
                        </div>
                      </div>
                      <div className="hive-stage-grid">
                        <div>
                          <span>Tahap 1 · byte asli</span>
                          <strong>{statusLabel(hive.stage1?.status)}</strong>
                          <small>
                            AI/deepfake dan metadata provider tidak memengaruhi
                            status visual.
                          </small>
                        </div>
                        <div>
                          <span>Tahap 2 · caption kanonis</span>
                          <strong>{statusLabel(hive.stage2?.status)}</strong>
                          <small>
                            {hive.stage2?.provider ||
                              "Hive VLM belum dijalankan"}
                          </small>
                        </div>
                        <div>
                          <span>Tahap 3 · preview normal</span>
                          <strong>
                            {statusLabel(
                              hive.stage3?.vlm?.status || hive.stage3?.status,
                            )}
                          </strong>
                          <small>
                            {hive.stage3?.models?.length || 0} output model V2 ·
                            observasi memerlukan telaah.
                          </small>
                        </div>
                      </div>
                      {!!hive.provider_status?.length && (
                        <details>
                          <summary>Status kapabilitas project key</summary>
                          <ul>
                            {hive.provider_status.map((item, index) => (
                              <li key={`${item.capability}-${index}`}>
                                {item.capability}: {statusLabel(item.status)}
                                {item.error_code ? ` (${item.error_code})` : ""}
                              </li>
                            ))}
                          </ul>
                        </details>
                      )}
                    </section>
                  )}
                  <div className="result-heading">
                    <div>
                      <div className="eyebrow">
                        HASIL PEMERIKSAAN{" "}
                        <span className="tiny-tag">
                          {bundle?.mode === "demo"
                            ? "DEMO / FIXTURE"
                            : extension?.method?.mode || "KOREKSI"}
                        </span>
                      </div>
                      <h2>
                        {analysis.atomic_claims.length} klaim atomik, jejak
                        bukti terbuka.
                      </h2>
                    </div>
                    <div className="export-buttons">
                      {["json", "zip", "csv", "overlay"].map((f) => (
                        <button
                          key={f}
                          onClick={() => void exportResult(f)}
                          title={`Ekspor ${f}`}
                        >
                          <ArrowDownToLine size={14} />
                          {f === "overlay" ? "Overlay" : f.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="summary-row">
                    {(
                      ["Supported", "Contradicted", "Unobservable"] as const
                    ).map((s) => (
                      <div key={s} className={`summary-stat ${s}`}>
                        <span className="status-symbol">
                          {s === "Supported" ? (
                            <Check size={16} />
                          ) : s === "Contradicted" ? (
                            <X size={16} />
                          ) : (
                            <CircleHelp size={16} />
                          )}
                        </span>
                        <strong>
                          {visual.filter((a) => a.visual_status === s).length}
                        </strong>
                        <span>{labels[s]}</span>
                      </div>
                    ))}
                    <div className="summary-stat timing">
                      <Clock3 size={16} />
                      <span>
                        {extension?.timing
                          ? `${(extension.timing.total_ms / 1000).toFixed(1)} detik`
                          : "Belum dianalisis"}
                      </span>
                    </div>
                  </div>
                  <div className="result-grid">
                    <div className="card evidence-card">
                      <div className="card-top">
                        <div>
                          <Focus size={17} />
                          <h3>Bukti pada gambar</h3>
                        </div>
                        <span className="tiny-tag">GRID REGIONS</span>
                      </div>
                      {bundle!.input.image && (
                        <MediaImage
                          media={bundle!.input.image}
                          assessments={visual}
                          selected={selected}
                        />
                      )}
                      <div className="evidence-hint">
                        <Focus size={15} />
                        {assessment?.supporting_regions.length ||
                        assessment?.contradicting_regions.length
                          ? "Region disorot untuk atom yang dipilih."
                          : "Belum ada region bukti untuk atom ini."}
                      </div>
                      <div className="original-caption">
                        <span className="field-label">
                          CAPTION TERSIMPAN · REVISI {bundle?.claim_revision}
                        </span>
                        <p>“{bundle?.input.claim_text}”</p>
                      </div>
                      <div className="evidence-note">
                        <ShieldCheck size={17} />
                        <p>
                          Grid adalah pembagian gambar, bukan segmentasi objek.
                          Token global hanya memberi konteks.
                        </p>
                      </div>
                    </div>
                    <div className="card atoms-card">
                      <div className="card-top">
                        <div>
                          <Layers3 size={17} />
                          <h3>Tahap 2 · klaim atomik</h3>
                        </div>
                        <button
                          className="text-button"
                          disabled={!!running}
                          onClick={() => {
                            setEditing(structuredClone(analysis.atomic_claims));
                            setReason("");
                          }}
                        >
                          <Pencil size={13} /> Koreksi atom
                        </button>
                      </div>
                      <div className="atom-list">
                        {analysis.atomic_claims.map((a, i) => {
                          const v = visual.find((v) => v.atom_id === a.atom_id);
                          return (
                            <button
                              key={a.atom_id}
                              className={`atom-row ${selected === a.atom_id ? "selected" : ""}`}
                              onClick={() => setSelected(a.atom_id)}
                            >
                              <span className="atom-number">
                                {String(i + 1).padStart(2, "0")}
                              </span>
                              <span className="atom-content">
                                <span className="atom-role">
                                  {roleNames[a.role]}
                                </span>
                                <strong>{a.statement}</strong>
                                <span
                                  className={`status-badge ${v?.visual_status || "pending"}`}
                                >
                                  {v
                                    ? labels[v.visual_status]
                                    : "Menunggu analisis ulang"}
                                </span>
                              </span>
                              <ChevronRight size={16} />
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                  {atom && (
                    <div className="card inspection">
                      <div className="inspection-title">
                        <span className="tiny-tag">{atom.atom_id}</span>
                        <h3>Dasar penilaian</h3>
                        <span>{roleNames[atom.role]}</span>
                      </div>
                      <p className="rationale">
                        {assessment?.rationale ||
                          "Atom telah dikoreksi. Hasil turunan dikosongkan sampai analisis ulang."}
                      </p>
                      {assessment?.counter_evidence && (
                        <div className="counter-evidence">
                          <strong>Observasi alternatif</strong>
                          <p>{assessment.counter_evidence}</p>
                        </div>
                      )}
                      <div className="inspection-metrics">
                        <div>
                          <span>Confidence parser</span>
                          <strong>{score(atom.parser_confidence)}</strong>
                        </div>
                        <div>
                          <span>Observabilitas</span>
                          <strong>
                            {score(assessment?.observability_score)}
                          </strong>
                        </div>
                        <div>
                          <span>Unmatched mass</span>
                          <strong>{score(assessment?.unmatched_mass)}</strong>
                        </div>
                        <div>
                          <span>Probabilitas head</span>
                          <strong>
                            {assessment?.probabilities
                              ? Object.entries(assessment.probabilities)
                                  .map(([l, p]) => `${l}: ${p.toFixed(3)}`)
                                  .join(" · ")
                              : "Tidak tersedia"}
                          </strong>
                        </div>
                      </div>
                      <details>
                        <summary>Struktur dan span caption</summary>
                        <p>
                          Subjek: {atom.subject ?? "—"} · Predikat:{" "}
                          {atom.predicate} · Objek: {atom.object ?? "—"}
                        </p>
                        <p>
                          Negasi: {atom.qualifiers.negated ? "Ya" : "Tidak"} ·
                          Kuantitas: {atom.qualifiers.quantity ?? "—"} · Waktu:{" "}
                          {atom.qualifiers.time ?? "—"} · Lokasi:{" "}
                          {atom.qualifiers.location ?? "—"}
                        </p>
                        <p>
                          Dependensi:{" "}
                          {atom.depends_on.join(", ") || "Tidak ada"}
                        </p>
                        {atom.spans.map((span, i) => (
                          <code key={i}>
                            [{span.start}, {span.end}){" "}
                            {[...bundle!.input.claim_text]
                              .slice(span.start, span.end)
                              .join("")}
                          </code>
                        ))}
                      </details>
                    </div>
                  )}
                  <div className="result-bottom">
                    <div className="card small-card">
                      <h3>
                        <ScanText size={16} /> Tahap 3 · multimodal
                      </h3>
                      <dl>
                        <dt>Region</dt>
                        <dd>{extension?.method?.region_method || "—"}</dd>
                        <dt>OCR</dt>
                        <dd>{analysis.ocr.length} observasi teks</dd>
                        <dt>Grounding</dt>
                        <dd>{extension?.method?.alignment || "—"}</dd>
                        <dt>Visual baseline</dt>
                        <dd>{extension?.method?.backbone || "—"}</dd>
                      </dl>
                    </div>
                    <div className="card small-card">
                      <h3>
                        <Settings2 size={16} /> Metode eksekusi
                      </h3>
                      <dl>
                        <dt>Backbone</dt>
                        <dd>
                          {extension?.method?.backbone || "Belum dijalankan"}
                        </dd>
                        <dt>Alignment</dt>
                        <dd>{extension?.method?.alignment || "—"}</dd>
                        <dt>Jenis inferensi</dt>
                        <dd>{extension?.method?.mode || "Koreksi manusia"}</dd>
                        <dt>Kalibrasi</dt>
                        <dd>Tidak tersedia</dd>
                        <dt>Checkpoint</dt>
                        <dd>{analysis.run.versions.head || "Tidak ada"}</dd>
                      </dl>
                    </div>
                    <div className="card small-card">
                      <h3>
                        <CircleHelp size={16} /> Catatan & keterbatasan
                      </h3>
                      <ul>
                        {[
                          ...(bundle?.warnings || []),
                          ...analysis.run.warnings,
                        ].map((w, i) => (
                          <li key={i}>{w.message}</li>
                        ))}
                        {!analysis.run.warnings.length && (
                          <li>
                            Hasil baru belum tersedia untuk atom yang dikoreksi.
                          </li>
                        )}
                      </ul>
                    </div>
                  </div>
                  <details className="card ocr-panel">
                    <summary>
                      Observasi OCR · {analysis.ocr.length} teks terbaca
                    </summary>
                    <p>
                      Tulisan pada gambar tidak membuktikan isi tulisan atau
                      tanggal pengambilan foto.
                    </p>
                    {analysis.ocr.map((o, i) => (
                      <div key={i}>
                        <code>{o.text}</code>{" "}
                        <span>
                          Skor OCR {score(o.confidence)} · {o.language}
                        </span>
                      </div>
                    ))}
                  </details>
                  <button
                    className="text-button"
                    onClick={() =>
                      void guard(async () => {
                        setSnapshots(
                          await api(`/api/v1/cases/${bundle!.case_id}/history`),
                        );
                        setShowSnapshots(!showSnapshots);
                      })
                    }
                  >
                    <History size={16} /> Lihat riwayat revisi kasus
                  </button>
                  {showSnapshots && (
                    <div className="card revision-history">
                      {snapshots.map((s, i) => (
                        <div key={i}>
                          <strong>{s.kind}</strong>
                          <span>
                            Revisi {s.bundle.claim_revision} ·{" "}
                            {s.bundle.analysis?.atom_set_id.slice(0, 20) ||
                              "Belum ada atom"}
                          </span>
                          <p>{s.reason}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              ) : (
                <div className="empty-guide">
                  <div>
                    <Layers3 size={23} />
                    <h3>Dari caption menjadi klaim</h3>
                    <p>
                      Aksi, atribut, waktu, dan lokasi dipisahkan agar dapat
                      dinilai secara mandiri.
                    </p>
                  </div>
                  <div>
                    <Focus size={23} />
                    <h3>Periksa bukti visualnya</h3>
                    <p>
                      Setiap dukungan atau bantahan memiliki region dan alasan
                      yang dapat diperiksa.
                    </p>
                  </div>
                  <div>
                    <CircleHelp size={23} />
                    <h3>Ketidakpastian tetap terlihat</h3>
                    <p>
                      Tidak terlihat bukan berarti salah. Cue yang kurang
                      menghasilkan Unobservable.
                    </p>
                  </div>
                </div>
              )}
              <div className="import-footer">
                <span>Punya hasil dari layanan AURORA lain?</span>
                <button
                  className="text-button"
                  onClick={() => importFile.current?.click()}
                >
                  <Upload size={14} /> Impor JSON / ZIP
                </button>
              </div>
            </>
          )}
          {page === "history" && (
            <section className="card history-page">
              <div className="card-top">
                <h2>
                  Semua kasus <span className="tiny-tag">{history.length}</span>
                </h2>
                <button
                  className="text-button"
                  onClick={() => importFile.current?.click()}
                >
                  <Upload size={15} /> Impor bundle
                </button>
              </div>
              {history.length ? (
                history.map((h) => (
                  <button
                    className="history-row"
                    key={h.case_id}
                    onClick={() => void loadCase(h.case_id)}
                  >
                    <span className="history-file">
                      <FileJson size={22} />
                    </span>
                    <span>
                      <strong>{h.caption}</strong>
                      <small>
                        {new Date(h.updated_at * 1000).toLocaleString("id-ID")}{" "}
                        · Revisi {h.claim_revision}
                      </small>
                    </span>
                    <span className="tiny-tag">{h.mode}</span>
                    <ChevronRight size={18} />
                  </button>
                ))
              ) : (
                <div className="empty-history">
                  <History size={36} />
                  <h3>Belum ada kasus tersimpan</h3>
                  <p>
                    Mulai dengan salah satu contoh demo atau unggah gambar Anda.
                  </p>
                  <button
                    className="button primary"
                    onClick={() => setPage("analysis")}
                  >
                    Mulai analisis <ArrowRight size={16} />
                  </button>
                </div>
              )}
            </section>
          )}
          {page === "methods" && (
            <>
              <div className="banner notice">
                <Activity size={18} />
                <span>{ready}</span>
              </div>
              <div className="capability-grid">
                {capabilities.map((c) => (
                  <div className="card capability-card" key={c.provider}>
                    <div>
                      <span className="tiny-tag">{c.mode}</span>
                      <span className={`cap-status ${c.status}`}>
                        {statusLabel(c.status)}
                      </span>
                    </div>
                    <h3>{c.provider}</h3>
                    <p>{c.message}</p>
                    <small>{c.capability}</small>
                  </div>
                ))}
              </div>
              <div className="card method-config">
                <h3>Konfigurasi pemeriksaan berikutnya</h3>
                <label>
                  Metode alignment
                  <select
                    value={alignment}
                    onChange={(e) => setAlignment(e.target.value)}
                  >
                    {[
                      "uot",
                      "balanced-ot",
                      "attention",
                      "max-region",
                      "mean-region",
                      "global",
                    ].map((m) => (
                      <option key={m}>{m}</option>
                    ))}
                  </select>
                </label>
                <p>
                  Probabilitas head dan skor alignment adalah besaran berbeda.
                  Kalibrasi utama serta verdict faktual menjadi tanggung jawab
                  modul keputusan.
                </p>
                <label>
                  Token akses server (jika diwajibkan)
                  <input
                    type="password"
                    autoComplete="off"
                    placeholder="Token hanya disimpan dalam sesi browser"
                    onChange={(e) =>
                      sessionStorage.setItem("aurora-token", e.target.value)
                    }
                  />
                </label>
                <button
                  className="button secondary"
                  onClick={() => void refresh()}
                >
                  Periksa koneksi
                </button>
              </div>
            </>
          )}
          <footer className="page-footer">
            <span>
              AURORA Visual <span>·</span> Konsistensi internal gambar–caption
            </span>
            <span>
              <ShieldCheck size={13} /> Dapat diaudit, tidak mengklaim
              kepastian.
            </span>
          </footer>
        </div>
      </main>
      <input
        ref={importFile}
        type="file"
        className="sr-only"
        accept=".json,.zip"
        aria-label="Impor bundle"
        onChange={(e) => {
          void importBundle(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {editing && (
        <div
          className="modal-backdrop"
          onKeyDown={(e) => {
            if (e.key === "Escape") setEditing(null);
          }}
        >
          <section
            ref={modalRef}
            className="editor-modal card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="editor-title"
          >
            <header>
              <div>
                <h2 id="editor-title">Koreksi klaim atomik</h2>
                <p>
                  Caption asli tetap tersimpan. Koreksi membuat atom set baru
                  dan mengosongkan hasil turunannya.
                </p>
              </div>
              <button
                aria-label="Tutup editor"
                onClick={() => setEditing(null)}
              >
                <X size={20} />
              </button>
            </header>
            <div className="editor-scroll">
              {editing.map((a, i) => (
                <div className="atom-editor" key={a.atom_id}>
                  <span className="tiny-tag">{a.atom_id}</span>
                  <label>
                    Proposisi
                    <input
                      value={a.statement}
                      onChange={(e) =>
                        setEditing(
                          editing.map((v, j) =>
                            j === i ? { ...v, statement: e.target.value } : v,
                          ),
                        )
                      }
                    />
                  </label>
                  <div className="field-row">
                    {(["subject", "predicate", "object"] as const).map((k) => (
                      <label key={k}>
                        {k === "subject"
                          ? "Subjek"
                          : k === "predicate"
                            ? "Predikat"
                            : "Objek"}
                        <input
                          value={a[k] ?? ""}
                          onChange={(e) =>
                            setEditing(
                              editing.map((v, j) =>
                                j === i
                                  ? {
                                      ...v,
                                      [k]:
                                        e.target.value ||
                                        (k === "predicate" ? "" : null),
                                    }
                                  : v,
                              ),
                            )
                          }
                        />
                      </label>
                    ))}
                  </div>
                  <div className="field-row">
                    <label>
                      Peran
                      <select
                        value={a.role}
                        onChange={(e) =>
                          setEditing(
                            editing.map((v, j) =>
                              j === i
                                ? { ...v, role: e.target.value as Atom["role"] }
                                : v,
                            ),
                          )
                        }
                      >
                        {Object.entries(roleNames).map(([k, v]) => (
                          <option value={k} key={k}>
                            {v}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={a.qualifiers.negated}
                        onChange={(e) =>
                          setEditing(
                            editing.map((v, j) =>
                              j === i
                                ? {
                                    ...v,
                                    qualifiers: {
                                      ...v.qualifiers,
                                      negated: e.target.checked,
                                    },
                                  }
                                : v,
                            ),
                          )
                        }
                      />{" "}
                      Negasi
                    </label>
                  </div>
                  <details>
                    <summary>Kualifikasi & referensi</summary>
                    <div className="field-row">
                      {(["quantity", "time", "location"] as const).map((k) => (
                        <label key={k}>
                          {k}
                          <input
                            type={k === "quantity" ? "number" : "text"}
                            value={a.qualifiers[k] ?? ""}
                            onChange={(e) =>
                              setEditing(
                                editing.map((v, j) =>
                                  j === i
                                    ? {
                                        ...v,
                                        qualifiers: {
                                          ...v.qualifiers,
                                          [k]: e.target.value
                                            ? k === "quantity"
                                              ? Number(e.target.value)
                                              : e.target.value
                                            : null,
                                        },
                                      }
                                    : v,
                                ),
                              )
                            }
                          />
                        </label>
                      ))}
                    </div>
                    <label>
                      Depends on (pisahkan koma)
                      <input
                        value={a.depends_on.join(",")}
                        onChange={(e) =>
                          setEditing(
                            editing.map((v, j) =>
                              j === i
                                ? {
                                    ...v,
                                    depends_on: e.target.value
                                      .split(",")
                                      .map((s) => s.trim())
                                      .filter(Boolean),
                                  }
                                : v,
                            ),
                          )
                        }
                      />
                    </label>
                    <p>
                      Span caption asli:{" "}
                      {a.spans.map((s) => `[${s.start}, ${s.end})`).join(", ")}.
                      Perubahan isi harus tetap sesuai span ini.
                    </p>
                  </details>
                </div>
              ))}
              <label>
                Alasan koreksi
                <textarea
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="Jelaskan bagian yang diperbaiki dan alasannya."
                />
              </label>
            </div>
            <footer>
              <button
                className="button secondary"
                onClick={() => setEditing(null)}
              >
                <ArrowLeft size={16} /> Batal
              </button>
              <button
                className="button primary"
                disabled={busy || !reason.trim()}
                onClick={() => void saveAtoms()}
              >
                <Check size={16} /> Simpan koreksi
              </button>
            </footer>
          </section>
        </div>
      )}
    </div>
  );
}
