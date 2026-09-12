# AURORA Visual

Modul 1 AURORA: aplikasi lokal untuk menguraikan caption Indonesia/Inggris menjadi klaim atomik, memeriksa konsistensi visual, dan mengekspor bukti yang dapat diaudit. Keluaran visual adalah **Supported / Contradicted / Unobservable**; aplikasi ini tidak menghasilkan verdict hoaks/fakta.

## Jalankan

Prasyarat: Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 22+, npm. OCR nyata memakai Tesseract dengan bahasa `ind` dan `eng` (`brew install tesseract tesseract-lang` di macOS; paket `tesseract-ocr-ind` dan `tesseract-ocr-eng` di Debian/Ubuntu). Tanpa Tesseract, analisis menjadi parsial dengan peringatan.

```sh
make setup
make dev
```

Buka [AURORA Visual](http://127.0.0.1:5171) dan [OpenAPI](http://127.0.0.1:8101/docs). `make demo` juga menjalankan aplikasi; pilih **Dukungan**, **Bantahan**, atau **Tak teramati**, lalu **Jalankan analisis**. Tidak memerlukan API key, GPU, atau unduhan bobot. Instalasi dependensi pertama tetap memerlukan internet. Setelah setup, jalur lokal tidak memakai layanan luar atau font eksternal.

Mode **Live lokal** menjalankan parser, grid, pengukuran warna nyata, OCR, dan UOT. Baseline lokal hanya mendukung klaim warna pada bidang yang seragam. Foto umum biasanya menghasilkan Unobservable; model ini tidak mengenali objek, identitas, afiliasi, jumlah, lokasi, atau relasi manusia. Kontradiksi membutuhkan warna alternatif terukur, bukan similarity rendah. `probabilities=null` untuk heuristic. OpenCLIP dan head terlatih merupakan adapter opsional; statusnya terlihat di halaman Metode.

## Alur 0–3 dan pemeriksaan asal media

Setiap analisis mengikuti empat tahap yang terlihat di UI:

1. **Tahap 0 — Masukan:** caption asli dan gambar PNG/JPG/WebP yang diunggah.
2. **Tahap 1 — Screening asal media:** nama field metadata, penanda perangkat lunak generatif yang dikenal, dan marker C2PA/JUMBF diperiksa secara lokal. Nilai asli EXIF tidak diekspor pada panel screening. Status AI-image, deepfake, dan watermark tak terlihat selalu **belum dikonfigurasi** sampai detektor tervalidasi disediakan.
3. **Tahap 2 — Urai klaim:** caption dipecah menjadi atom aktor, aksi, objek, atribut, lokasi, waktu, jumlah, relasi, atau sebab.
4. **Tahap 3 — Analisis multimodal:** OCR, grid region, alignment, dan assessment visual konservatif dijalankan.

Data tahap 1 disimpan secara lokal di `extensions.aurora_visual.screening`; kontrak publik AURORA 1.0.0 tidak berubah. Screening asal media bukan validasi C2PA kriptografis, bukan detektor umum AI/deepfake/watermark, tidak mengubah status **Supported / Contradicted / Unobservable**, dan tidak menentukan kebenaran caption.

## Diagnostik MAFINDO opsional

MAFINDO V1 tersedia hanya sebagai uji kompatibilitas provider baca-saja. Hasilnya tidak masuk ke `retrieval`, `decision`, assessment visual, atau ekspor verdict AURORA. Simpan credential hanya pada konfigurasi server lokal, tidak pernah pada `VITE_*`, frontend, bundle, atau fixture:

```sh
AURORA_MAFINDO_API_KEY=... make mafindo-live-test
```

Perintah tersebut meminta endpoint `latest(1)`, menyembunyikan URL/credential dari output, dan tidak dijalankan oleh `make test`, `make smoke`, atau `make browser-test`. Tanpa environment variable tersebut, perintah berhenti sebelum menghubungi provider.


| Perintah | Hasil |
|---|---|
| `make test` | Lint, format, TypeScript, golden JCS lintas bahasa, unit/contract/integration tests |
| `make smoke` | Server + database baru, worker proses terpisah, fixture S/C/U, live lokal, semua ekspor, restart dan idempotency |
| `make browser-test` | Instal browser pengujian bila belum tersedia, lalu uji desktop/ponsel dengan database terpisah |
| `make evaluate` | Training CPU fixture 3 epoch dan metrik/intervensi fitur pada test fixture |
| `make experiments` | Training ulang 11 baseline/ablation/probe pada satu seed smoke |
| `make build` | Build frontend produksi |
| `make schema` | JSON Schema, OpenAPI dan fixture kanonis |

Hasil aktual disimpan di `artifacts/reports/`; JSON/ZIP/CSV/overlay untuk serah terima di `artifacts/handoff/`. Screenshot browser desktop/ponsel dan catatan smoke browser disertakan. Metrik fixture adalah pemeriksaan perangkat lunak, **bukan hasil penelitian**. Lihat [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

CLI tambahan: `uv run aurora --help` menyediakan `train`, `resume`, `predict`, `evaluate`, `cache-features`, `export-checkpoint`, `experiments`, `validate`, `schema`, `weak-supervision`, dan `import-dataset`. Contoh:

```sh
uv run aurora resume --smoke --epochs 5 --checkpoint artifacts/checkpoints/smoke.pt --output artifacts/checkpoints/resumed.pt
uv run aurora predict --smoke --checkpoint artifacts/checkpoints/smoke.pt --output artifacts/reports/predictions.json
uv run aurora export-checkpoint artifacts/checkpoints/smoke.pt --output artifacts/checkpoints/smoke.safetensors
uv run aurora evaluate-parser --output artifacts/reports/parser-benchmark.json
uv run aurora robustness --smoke --checkpoint artifacts/checkpoints/smoke.pt --output artifacts/reports/robustness-smoke.json
```

`predict` dan `evaluate` mengikuti metode alignment yang disimpan dalam checkpoint kecuali `--method` diberikan secara eksplisit. Keduanya menolak fitur dengan encoder/preprocessing/grid yang tidak cocok. Evaluasi parser mengembalikan status menunggu review manusia sampai gold benchmark tersedia. Robustness `--smoke` menguji jalur perturbasi piksel dengan checkpoint fixture; perbedaan fitur sintetis diberi status `fixture_override_unvalidated`, tanpa klaim mutu model.

## Data dan konfigurasi

Salin `.env.example` menjadi `.env` jika diperlukan; jangan masukkan secret ke frontend. `var/aurora.db` menyimpan kasus, snapshot revisi, audit dan job; `var/media`, `var/derived`, `var/cache`, `var/artifacts` menyimpan berkas. Jangan menghapus folder ini untuk melakukan restart.

`AURORA_OPENCLIP_PRETRAINED` menerima checkpoint lokal atau nama pretrained yang didukung OpenCLIP. Nama remote mengizinkan unduhan bobot saat fitur dipilih; verifikasi lisensi bobot dan ruang disk sebelum mengaktifkannya. `AURORA_OPENCLIP_MODEL` default `ViT-B-32`; model beku, tidak menggunakan random weights sebagai hasil. `AURORA_CHECKPOINT` hanya untuk checkpoint head dengan metadata lengkap dan `data_kind=research`; checkpoint smoke ditolak pada live. Dukungan Indonesia OpenCLIP standar belum divalidasi.

## Docker dan satu server

```sh
docker compose up --build
```

Compose mengikat port API 8101 dan frontend 5171 ke **127.0.0.1**, dengan volume data persisten. Dockerfile/Compose disediakan; runtime Docker belum tersedia pada mesin implementasi sehingga build container belum diverifikasi. Dependensi PyTorch pada image dapat berukuran besar. Untuk satu server, baca [docs/deployment.md](docs/deployment.md): TLS/reverse proxy, token autentikasi, pembatasan request, dan backup wajib diatur sebelum ekspos publik. Tidak ada deployment publik yang dilakukan.

Panduan rinci: [arsitektur](docs/architecture.md), [API](docs/api.md), [data](docs/data.md), [evaluasi](docs/evaluation.md), [batasan](docs/limitations.md), [handoff](docs/handoff.md).
