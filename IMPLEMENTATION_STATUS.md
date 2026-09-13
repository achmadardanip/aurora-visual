# Status implementasi AURORA Visual

Diperbarui **13 September 2026**. Modul 1 dan kontrak **v1.0.0**. Implementasi lokal, demo, dan jalur komputasi nyata yang terbatas telah diverifikasi. **Belum ada hasil evaluasi penelitian atau model untuk menilai foto berita secara umum.** Pekerjaan yang tersimpan dari sesi sebelumnya dipertahankan.

## Status per kemampuan

| Bagian | Status | Bukti dan batas |
| --- | --- | --- |
| API, worker, database | implemented, demo_verified | FastAPI, SQLAlchemy/SQLite, migrasi Alembic, antrean persisten, lease/timeout/recovery, idempotency dan snapshot revisi. Smoke menjalankan server dan proses worker dengan database sementara baru, kemudian restart. |
| Frontend Indonesia | implemented, demo_verified | Alur terlihat Tahap 0–3: unggah/caption, screening asal media, atom, lalu analisis multimodal. Demo S/C/U, progres, hasil per atom, sorotan region, koreksi, riwayat, kemampuan model, impor/ekspor diuji pada desktop 1280 px dan ponsel 390 px. |
| Screening asal media (Tahap 1) | implemented, live_verified terbatas | Memeriksa byte asli, nama/presence metadata dan marker C2PA/JUMBF; output hanya di `extensions.aurora_visual.screening`. Bukan validator C2PA atau detektor AI/deepfake/watermark tervalidasi, dan tidak mengubah S/C/U atau kebenaran caption. |
| MAFINDO V1 diagnostik | implemented, live_verified | Bridge baca-saja server-side dengan host tetap, encoding path, batas respons dan redaksi secret. Uji provider 13 September 2026 menerima 20/20 record ternormalisasi (20 ID unik/judul/tanggal valid; rentang 3–12 September 2026) dan uji kompatibilitas `latest(1)` lulus. Output tidak disimpan atau diterjemahkan ke bundle, retrieval, decision, assessment visual, atau verdict; endpoint dinonaktifkan pada public mode. MAFINDO belum terbukti menyediakan corpus visual berlisensi untuk training. |
| Kontrak dan pertukaran | implemented, demo_verified | Pydantic, JSON Schema, OpenAPI, TypeScript, hash JCS Python/JavaScript, Unicode/emoji, contoh kontrak dan guard hasil downstream. ZIP berisi media/sidecar; JSON, CSV dan overlay tersedia. Validator mandiri repo modul 2/3 belum dijalankan karena repo tersebut tidak tersedia. |
| Parser aturan ID/EN | implemented, live_verified terbatas | Parser nyata tanpa API key; mempertahankan SPO, negasi, angka, peran, span code point, waktu/lokasi dan warning ambiguitas. Bukan dependency/SRL/NER penuh. Benchmark manusia belum ditinjau. |
| Gambar dan region | implemented, live_verified terbatas | Hash byte asli, EXIF orientation, RGB, grid 1–32 region, global context, cache fitur. Grid bukan segmentasi objek. |
| OCR | implemented, live_verified terbatas | Tesseract `ind+eng` tersedia dan diuji pada gambar uji. Teks yang terbaca tidak membuktikan tanggal foto atau kebenaran isi tulisan. |
| Baseline lokal | implemented, live_verified terbatas | Pengukuran warna bidang nyata → alignment → heuristic S/C/U. `probabilities=null`; selain cue warna bidang yang cukup, kesimpulan konservatif U. Tidak mengenali objek, orang, lokasi, relasi, atau jumlah. |
| Alignment dan UOT | implemented, demo_verified | Global/max/mean/attention/balanced OT/UOT; generalized KL, Sinkhorn log-domain, diagnostics, unmatched mass, uji pembanding SciPy dan gradien projector/cost. UOT juga dijalankan pada jalur lokal nyata. |
| Training dan head | implemented, demo_verified | Class-balanced CE, contrastive/margin, hard-positive consistency, entropy; save/load/resume/predict/export-checkpoint diuji CPU. Checkpoint fixture ditolak untuk live/confirmatory. |
| Evaluasi dan eksperimen | implemented, demo_verified | 11 baseline/ablation/probe, metrik atom/role, intervensi fitur, group bootstrap, robustness piksel, parser benchmark runner. Lima seed dikonfigurasi; eksekusi di sini satu seed smoke. |
| OpenCLIP beku | implemented; live belum diverifikasi | Adapter, cache dan deteksi CUDA/MPS/CPU tersedia. Checkpoint pretrained belum dipilih/diunduh/dijalankan. Bahasa Indonesia belum divalidasi. |
| Parser LLM opsional | implemented; live belum diverifikasi | Adapter structured output dan validasi tersedia; endpoint/model Ollama belum dikonfigurasi. |
| Dataset publik/importer | implemented; research_evaluated=false | Adapter NewsCLIPpings, VERITE, COSMOS, MMFakeBench serta pemeriksaan sumber primer tersedia. Label post-level tidak menjadi gold atom. Dataset berlisensi dan anotasi manusia belum tersedia. |
| Docker/deployment | single-server hardened; container_verified | Build API/frontend dan smoke Compose arm64 lulus pada Docker Desktop 29.7.2/Compose 5.5.1. Public mode mewajibkan token 32+ karakter, exact CORS dan trusted host; MAFINDO diagnostik dinonaktifkan, header keamanan/HSTS aktif, API non-root, dan backup/restore SQLite-media tersedia. Domain/TLS/Caddy runtime/monitoring belum dideploy; bukan arsitektur multi-user/scale-out. |

## Bukti pemeriksaan aktual

Ringkasan mesin, perintah, exit code, dan lokasi log tersedia di [verification.json](artifacts/reports/verification.json) serta laporan terkait. Pemeriksaan Stage 0–3 berikut menambahkan cakupan screening dan isolasi MAFINDO tanpa mengubah kontrak publik. Angka lulus terbaru berada di `pytest.xml`; `checks.log` adalah snapshot run terdahulu dan tidak menjadi sumber hitungan terbaru.

| Perintah/pemeriksaan | Hasil aktual | Artefak |
| --- | --- | --- |
| `make setup` | Lulus; dependency terkunci, frontend terpasang, migrasi/schema selesai. | [setup.log](artifacts/reports/setup.log) |
| `make test` | Ruff, format, TypeScript, Prettier dan golden JCS lulus; **112 passed**, 0 failed, 3 warning yang telah diketahui. | [pytest.xml](artifacts/reports/pytest.xml); [checks.log](artifacts/reports/checks.log) adalah snapshot run sebelumnya. |
| `make smoke` | Lulus: fixture S/C/U tetap semantik sama, live warna lokal, ekspor JSON/ZIP/CSV/overlay, worker terpisah, restart, dan idempotency setelah restart. | [http-smoke.json](artifacts/reports/http-smoke.json), [http-smoke.log](artifacts/reports/http-smoke.log) |
| `make browser-test` | **2 passed**, 0 skipped/flaky: desktop memeriksa Stage 1/3, koreksi atom → invalidasi → analisis ulang → reload/riwayat; ponsel memeriksa Stage 1 dan tanpa overflow. | [playwright.json](artifacts/reports/playwright.json), [browser-smoke.log](artifacts/reports/browser-smoke.log) |
| `make build` | Build produksi frontend lulus. | [build.log](artifacts/reports/build.log) |
| `make evaluate` | Training fixture CPU 3 epoch/15 steps, loss finite, 0 nonconverged batch pada epoch terakhir; evaluasi test fixture selesai. | [training.log](artifacts/reports/training.log), [evaluation-smoke.json](artifacts/reports/evaluation-smoke.json) |
| `make experiments` | 11 kondisi selesai, seed 17, masing-masing 2 epoch. | [ablations-smoke.json](artifacts/reports/ablations-smoke.json), [experiments.log](artifacts/reports/experiments.log) |
| `aurora resume --smoke --epochs 5 …` | Lulus melanjutkan checkpoint 3 → 5 epoch, 25 total steps. | [resume.log](artifacts/reports/resume.log), `artifacts/checkpoints/resumed.pt` |
| `aurora predict --smoke …` | Lulus; probabilitas dan metode checkpoint tercatat. | [predictions.json](artifacts/reports/predictions.json) |
| `aurora export-checkpoint …` | Bobot safetensors dan metadata/hash diekspor. | `artifacts/checkpoints/smoke.safetensors`, [export-checkpoint.log](artifacts/reports/export-checkpoint.log) |
| `aurora cache-features --smoke …` | CLI fixture 8 sampel lulus; cache fitur piksel nyata juga dipakai smoke HTTP/robustness. | [cache-smoke.json](artifacts/reports/cache-smoke.json) |
| `aurora robustness --smoke …` | Original, blur, crop, OCR overlay selesai; override fitur fixture ditandai eksplisit. | [robustness-smoke.json](artifacts/reports/robustness-smoke.json) |
| `aurora evaluate-parser …` | Runner lulus; **awaiting_human_review**, 12 kandidat, 0 reviewed, metrics=null. | [parser-benchmark.json](artifacts/reports/parser-benchmark.json) |
| `make mafindo-live-test` dengan key server-side | Lulus pada 13 September 2026: `latest(1)` kompatibel (~781 ms). Uji sampel terpisah meminta/menerima 20 record; semua ID unik, judul terisi, dan tanggal dapat diparse. Data tidak dipersistenkan atau dipakai untuk training/verdict. | Output terminal terkontrol; credential dan record mentah tidak ditulis ke artefak. |
| Backup create/verify/restore | Lulus: SQLite backup API, staging media/sidecar, schema manifest/checksum SHA-256, batas dan validasi member archive, `PRAGMA integrity_check`, serta restore atomik ke directory kosong disposable; WAL/SHM transien tidak dimasukkan. | CLI `make backup`, `make verify-backup ARCHIVE=…`, `make restore-backup ARCHIVE=… RESTORE_DIR=…`; unit test backup/restore. |
| Docker image/Compose | Lulus pada Docker Desktop 29.7.2/Compose 5.5.1 arm64: kedua image dibangun, migrasi `0001`, API UID/GID 10001, volume writable, SQLite integrity, health/readiness langsung/proxy, syntax Nginx, browser Tahap 0–3 origin container, restart kasus/media + replay idempotency, public auth/Host/CORS/HSTS/MAFINDO-disablement, dan drill backup/verify/restore named volume. CLI backup tersedia dalam image; filesystem image tidak memuat `.env`, `.claude`, atau `prompt-aurora`; Uvicorn tanpa header `Server`. Pemindaian CVE Docker Scout tertunda karena login Docker diperlukan. | Smoke container disposable 13 September 2026; stack dan volume uji telah dihapus. Caddy/TLS/domain publik belum diuji runtime. |

Screenshot aktual: [desktop](artifacts/reports/browser-desktop.png), [ponsel](artifacts/reports/browser-mobile.png). Skrip browser memakai database sementara sendiri, sehingga tidak menghapus atau mengubah riwayat pengguna.

Penyesuaian Stage 1 dan MAFINDO menambahkan pengujian unit/API/UI; format Ruff/Prettier telah diperbaiki sebelum seluruh pemeriksaan diulang. Tiga warning pada pytest berasal dari dua deprecation dependency dan pembuatan ZIP duplikat yang sengaja dipakai tes penolakan impor.

## Interpretasi evaluasi

Evaluasi fixture memuat **3 atom dari 1 event test sintetis**. Macro-F1=1.0, contradiction F1=1.0 dan false-contradiction rate=0.0 hanya membuktikan jalur perangkat lunak pada fixture kecil. CI tetap null karena hanya ada satu unit independen. Mean inference yang terukur sekitar 1.82 ms dan peak RSS proses 253,657,088 byte pada run CPU ini; bukan benchmark GPU atau kemampuan foto nyata. Tidak ada angka yang boleh dipakai sebagai hasil skripsi.

Robustness smoke menerapkan checkpoint yang dilatih pada tensor sintetis ke descriptor piksel untuk memeriksa wiring; setiap variasi diberi `feature_compatibility=fixture_override_unvalidated`. `gold_robustness_metrics=null`. Jalur manifest/checkpoint penelitian menolak ketidakcocokan fitur. Skor alignment dan penghapusan/penambahan embedding bukan bukti faithfulness kausal pada piksel.

## Perbaikan pada sesi lanjutan

- Pengujian browser yang sebelumnya gagal karena browser pengujian belum tersedia kini lulus dengan `make browser-test`.
- `predict`/`evaluate` memeriksa encoder/preprocessing/grid terhadap checkpoint, termasuk bila dimensi tensor tetap sama. Default alignment mengikuti checkpoint, dan override eksplisit tetap didukung serta dicatat.
- Robustness memakai metode checkpoint dan menolak fitur yang tidak cocok pada jalur penelitian; pengecualian fixture smoke dinyatakan dalam output.
- Lima tes regresi ditambahkan untuk perilaku tersebut. Panduan browser, evaluasi, konfigurasi proxy lokal dan dokumen status ini dilengkapi.

## Cara menjalankan

```sh
make setup
make dev
```

Buka [UI lokal](http://127.0.0.1:5171) dan [OpenAPI](http://127.0.0.1:8101/docs). `make demo` juga menjalankan aplikasi. Pilih **Dukungan**, **Bantahan**, atau **Tak teramati**, lalu **Jalankan analisis**. Tidak membutuhkan API key/GPU atau bobot besar. Setup dependency dan pemasangan browser pertama memerlukan internet.

Pada pemeriksaan akhir, API lokal memberikan `/health` **200 ok** dan `/ready` **200 degraded**: database, worker, demo, warna lokal, serta OCR siap; OpenCLIP, head penelitian, dan LLM belum dikonfigurasi. Status degraded berasal dari kemampuan opsional tersebut.

Lingkungan aktual: macOS 27.0 arm64, Python 3.12.13, Node 26.0.0, npm 11.12.1, uv 0.12.8, PyTorch 2.14.0. MPS tersedia, CUDA tidak; training/evaluasi smoke memakai CPU. Tesseract memiliki bahasa `ind` dan `eng`. Repository Git dipublikasikan pada `main` di commit `1a7b0523b53f3c581f94f246b775ea50a630d382`; hardening produksi saat ini berada di branch `production-hardening` dan belum dipush.

## Yang perlu disediakan untuk jalur berikutnya

| Konfigurasi/aset | Kebutuhan |
| --- | --- |
| Demo dan baseline warna lokal | Tidak perlu secret atau model tambahan setelah setup; OCR memerlukan Tesseract `ind+eng`. |
| `AURORA_OPENCLIP_MODEL`, `AURORA_OPENCLIP_PRETRAINED` | Pilih bobot dengan model card/lisensi yang sesuai. Aktifkan pilihan OpenCLIP pada UI atau set `AURORA_BACKBONE=openclip`. Nama pretrained dapat memicu unduhan; checkpoint lokal juga didukung. |
| `AURORA_CHECKPOINT` | Checkpoint head penelitian beserta metadata JSON, split/hash dan identitas fitur yang cocok; checkpoint smoke tidak dapat dipakai live. |
| `AURORA_LLM_URL`, `AURORA_LLM_MODEL`, `AURORA_LLM_ALLOWED_ORIGINS` | Server Ollama-compatible opsional, model tersedia, dan origin dalam allowlist server. Belum diuji live. |
| Dataset/anotasi | Media dengan hak akses/lisensi yang sesuai, grouping image/event/source, label atom/region dan dua anotator manusia. Kandidat benchmark belum menjadi gold. |
| Eksperimen penelitian | Jalankan lima seed, perbandingan parser, lintas dataset, kalibrasi yang relevan dan profiling GPU setelah data/model tersedia. Kalibrasi/fusi faktual utama dimiliki modul 3. |
| Deployment server | Pilih server/domain, atur DNS/TLS/proxy/firewall, `AURORA_PUBLIC=true`, token acak 32+ karakter, exact `AURORA_CORS`/`AURORA_ALLOWED_HOSTS`, backup terenkripsi off-host, restore drill, monitoring, reboot/recovery, dan acceptance test eksternal; ikuti [deployment.md](docs/deployment.md). Build dan smoke Compose lokal arm64 sudah lulus, tetapi ini tidak menggantikan verifikasi host produksi. Tidak diperlukan untuk localhost. |

FG-CLIP/SAM2/TextRegion belum diaktifkan. Kinerja target GPU 16–24 GB, akurasi umum, benchmark manusia dan target peningkatan ilmiah **belum diverifikasi**.

## Serah terima modul 2/3

- Definisi bersama: [Pydantic](backend/app/models/contract.py), [JSON Schema](contracts/aurora.schema.json), [OpenAPI](contracts/openapi.json), [TypeScript](contracts/aurora.ts). Dokumen prompt internal yang menjadi acuan awal sengaja tidak dipublikasikan.
- Golden lintas bahasa: [golden-bundle.json](contracts/golden-bundle.json), [jcs-golden.json](contracts/jcs-golden.json).
- Output aktual: [supported.zip](artifacts/handoff/supported.zip), [contradicted.zip](artifacts/handoff/contradicted.zip), [unobservable.zip](artifacts/handoff/unobservable.zip). JSON/CSV/overlay pendamping berada di folder yang sama, bersama [live-local.json](artifacts/handoff/live-local.json).
- Panduan konsumsi atom set, probability nullable, sidecar, media dan run provenance: [docs/handoff.md](docs/handoff.md).

Semua fixture/demo tetap berlabel. Output visual tidak menghasilkan verdict hoaks/fakta, retrieval, atau forensic signal. Kedua repo penerima perlu menjalankan validatornya sendiri pada bundle ekspor sebelum integrasi tim dinyatakan selesai.
