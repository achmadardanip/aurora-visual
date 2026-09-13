# Arsitektur

Aplikasi memakai FastAPI/Pydantic v2, SQLAlchemy/SQLite WAL, worker persisten, PyTorch, dan React/TypeScript/Vite. Frontend hanya mengakses service API; tidak memiliki jalur inferensi sendiri. Paket prompt asli tetap dipertahankan.

```
PNG/JPEG/WebP + caption asli
   → hash byte asli / EXIF transpose / RGB turunan
   → screening asal media lokal (field metadata + validasi manifest/tanda tangan C2PA)
   → Atomizer ID/EN atau atom hasil koreksi
   → grid + global token / frozen features / Tesseract
   → global/max/mean/attention/balanced OT/UOT
   → heuristic konservatif atau checkpoint head tervalidasi
   → Analysis + sidecar / snapshot database / audit
   → UI atom–region / JSON / ZIP + media / CSV / overlay
```

### Tahap 0–3

Tahap 0 menerima caption asli dan byte media. Tahap 1 menjalankan `aurora_visual.origin_screen.screen_image` terhadap byte asli sebelum preview RGB; outputnya memakai nama field metadata, klasifikasi perangkat lunak, dan — melalui c2pa-python — penguraian manifest C2PA beserta validasi tanda tangan terhadap trust store bawaan SDK (trust anchor tambahan via `AURORA_C2PA_TRUST_ANCHORS`). Hanya manifest tervalidasi dengan digitalSourceType algoritmik terlatih yang dapat menaikkan label `likely_ai_generated`; manifest tidak tervalidasi tidak dipercaya. Hasil tersimpan pada `extensions.aurora_visual.screening`, bersama target `asset_id`/SHA-256, keterbatasan, dan penanda tegas bahwa hasil tidak memengaruhi assessment visual atau kebenaran klaim. Deteksi AI-image/deepfake Hive V3 (opt-in, mode live) dan deteksi watermark SynthID (`aurora_visual.synthid`, gateway operator, opt-in per analisis) dilaporkan terpisah dan tidak pernah menjadi validator autentisitas.

Tahap 2 mengurai caption menjadi atom. Tahap 3 menjalankan OCR, region grid, fitur visual, alignment, dan assessment konservatif. `screening` tidak dimasukkan ke `Retrieval.forensic_signals`, `retrieval`, atau `decision`; rerun analysis juga tetap mengosongkan retrieval/decision seperti sebelumnya.

### Diagnostik provider MAFINDO

`aurora_visual.mafindo.MafindoClient` adalah bridge diagnostik server-side baca-saja dengan host tetap dan encoding setiap segmen path. Kunci hanya dibaca dari `AURORA_MAFINDO_API_KEY`; response provider dibatasi dan dinormalisasi, serta pesan kegagalan tidak memuat URL atau credential. Endpoint `GET /api/v1/providers/mafindo/latest` dan `scripts/mafindo_live_test.py` tidak mengikat hasil ke kasus atau bundle. Provider status/klasifikasi tidak diterjemahkan menjadi label AURORA.

`backend/app/api` menangani transport dan autentikasi, `services` aturan kasus/media/portabilitas/pipeline, `models` kontrak dan persistence, `workers` lease dan proses komputasi. `backend/aurora_visual` memisahkan atomization, vision, OCR, alignment, entailment, training dan evaluation. Kontrak evidence/forensics/decision tersedia untuk validasi dan pertukaran, tanpa mengimplementasikan retrieval atau fusi milik modul 2/3.

## Identitas dan snapshot

Hash SHA-256 media memakai byte unggahan asli; gambar RGB pasca-EXIF merupakan turunan terpisah. Bbox relatif terhadap turunan itu. URI API relatif dan URI ZIP dapat berubah tanpa mengubah asset_id. JCS memakai library RFC 8785, bukan sekadar JSON sort_keys. Hash atom mencakup seluruh metadata Atom sesuai kontrak. Golden vector Unicode/emoji/desimal memeriksa kesamaan byte dan hash JavaScript/Python.

Case menyimpan snapshot terkini; tabel snapshots menyimpan input, output run, impor dan koreksi. Koreksi atom menjaga claim_revision dan caption, membuat atom_set_id baru, mencatat parent dan alasan, serta membuang hasil visual/retrieval/decision terkini. Perubahan gambar/caption wajib revision+1. Run mencatat hash snapshot dan parent run IDs. Job yang selesai sesudah snapshot berubah ditolak sebagai STALE_RESULT.

## Worker

Satu supervisor default berada di proses API, satu child process per pekerjaan. SQLite `BEGIN IMMEDIATE` mengklaim job secara atomik. Job, key idempotency, hash request, run_id, attempts, progress dan lease tersimpan di database. Lease = timeout+15 detik; setelah restart job kedaluwarsa direqueue, maksimal tiga attempts. Supervisor mengirim heartbeat, membatasi waktu child, dan mendukung pembatalan. Shutdown mematikan child dan membiarkan pemulihan lease; bukan task HTTP panjang yang hilang tanpa jejak.

Batas default: 20 queued/running jobs per pemilik, 180 detik/job, satu komputasi pada satu mesin. Model/fitur besar dapat memerlukan timeout yang dikonfigurasi. Multi-server belum menjadi target: migrasikan SQLite ke PostgreSQL, ganti klaim job dengan `FOR UPDATE SKIP LOCKED`, gunakan identitas worker unik serta artifact storage bersama sebelum menambah worker lintas mesin.

## Representasi dan penalaran

Grid default 4×4 (16 region), maksimum 32. Grid tidak diklaim sebagai proposal objek dan confidence proposal null. Fitur lokal 12 dimensi = proporsi enam warna, mean RGB, std RGB. Descriptor teks lokal hanya kosakata warna ID/EN; tidak dianggap embedding semantik umum. Encoder OpenCLIP terpisah menggunakan `eval()`, `requires_grad_(False)` dan `inference_mode`, dengan perangkat CUDA/MPS/CPU. Global image dan full-caption feature disimpan terpisah; global token tidak diklaim sebagai grounding.

Cache kunci SHA media + teks persis + backbone/checkpoint + preprocessing/config/bahasa. Checkpoint lokal dihitung hash; berkas NPZ dimuat dengan `allow_pickle=False`. OCR Tesseract memakai `ind+eng`, dengan bbox, bahasa, skor engine, timeout 20 detik, dan peringatan bahwa tulisan bukan tanggal pengambilan foto.

Heuristic hanya memberi dukungan/bantahan warna bila subjek secara eksplisit bidang/latar/gambar/kanvas dan **semua** grid memiliki kemurnian warna ≥0.96. Negasi dipertahankan. Klaim objek merah tidak bisa dibantah hanya karena latar biru. Kemurnian adalah observabilitas operasional, bukan probabilitas terkalibrasi. Identitas, tanggal, lokasi, sebab, jumlah dan relasi tetap U tanpa pengamat khusus. Jika head aktif, distribusi/logit disimpan; gate bukti eksplisit mencegah model mengarang kontradiksi atau dukungan ketika cue tidak memadai.

## UOT dan head

Solver log-domain meminimalkan `<Π,C> + ε KL(Π||abᵀ) + τt KL(Π1||a) + τv KL(Πᵀ1||b)`. KL generalized mencakup `-p+q`. `logK=loga+logb-C/ε`, update scaling memakai faktor `τ/(τ+ε)`. Priors positif, finite checks, tolerance, iteration cap dan objective diagnostics tersedia. Row/column tidak dipaksa sama dengan prior pada UOT. ρ = clamp(a-rowmass,0)/a. Balanced OT memakai faktor 1. Tes membandingkan objective/coupling ke SciPy L-BFGS-B constrained dan limit τ besar, serta memeriksa gradien projector/cost.

Projector atom menerima embedding + 9 role indicators + negasi/kuantitas/kualifikasi/quality parser. Projector region menerima embedding + global context + bbox/quality. Cost adalah kombinasi nonnegatif ternormalisasi softmax atas cosine, role compatibility, attribute distance, dan relation geometry. Relation cost adalah **aproksimasi unary bersyarat anchor**: first-pass cosine memberi center region ekspektasian untuk dependency atom; offset kiri/kanan/atas/bawah memprediksi center relatif. Cost hanya diaktifkan untuk atom relation dengan dependency. Bukan full graph OT, bukan detektor relasi manusia.

Head menerima atom, coupling-weighted region, min cost, entropy, unmatched mass, quality parser/proposal, serta cue bantahan eksplisit. Probabilitas head tidak sama dengan skor alignment dan belum dikalibrasi. Coupling, cost, regional/global features, skor dan logit berada di sidecar bernama berdasarkan run. Pipeline UOT heuristic memakai cosine-only yang benar-benar dihitung; learned cost aktif pada training/checkpoint, tidak random initialization untuk pengguna.
