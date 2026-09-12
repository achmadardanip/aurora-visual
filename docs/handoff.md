# Serah terima modul 1 ke modul 2/3

Kontrak **1.0.0** tetap sama dengan `prompt-aurora/KONTRAK_BERSAMA.md`. Gunakan:

- `contracts/aurora.ts`: tipe normatif yang disalin dari kontrak, seluruh interface diekspor.
- `contracts/aurora.schema.json`: schema Pydantic, field required/nullable eksplisit.
- `contracts/openapi.json`: endpoint dan schema envelope.
- `contracts/golden-bundle.json`, `contracts/jcs-golden.json`: Unicode/emoji/null/desimal, byte kanonis, hash dan atom set.
- `artifacts/handoff/supported.*`, `contradicted.*`, `unobservable.*`: JSON/ZIP/CSV/overlay aktual dari clean HTTP smoke.
- `artifacts/handoff/live-local.json`: pipeline lokal nyata heuristic; bukan pretrained VLM.

## Screening asal media (Tahap 1)

Analisis dapat menyertakan `extensions.aurora_visual.screening`. Ini ekstensi lokal Modul 1, bukan tambahan pada `Retrieval.forensic_signals` atau kontrak publik AURORA 1.0.0. Objek ini mencatat target asset, nama/presence field metadata, klasifikasi marker perangkat lunak, dan marker byte C2PA/JUMBF tanpa mengekspor nilai EXIF mentah.

`marker_present` hanya berarti byte marker ditemukan; `verification=not_verified` bukan parsing manifest, verifikasi signature, atau trust-chain C2PA. Status detektor `unavailable` berarti tidak dinilai, bukan hasil negatif. Consumer harus mempertahankan batas ini: screening tidak boleh diterjemahkan menjadi Supported/Contradicted/Unobservable, forensic signal, retrieval, decision, atau verdict fakta.

MAFINDO V1 adalah bridge diagnostik server-side terpisah. Respons provider tidak pernah ditulis ke bundle, snapshot kasus, sidecar, `retrieval`, `decision`, atau assessment visual. Modul 2/3 tidak boleh menganggap endpoint diagnostik atau klasifikasi provider sebagai evidence AURORA tanpa perubahan kontrak dan desain bersama yang terdokumentasi.

Modul 2 menerima bundle output analyze, memakai atom_set_id yang sama dan dapat menambahkan retrieval; modul 3 menambahkan decision setelah analysis valid. Modul 1 tidak menghasilkan evidence, forensic signals atau verdict faktual. Output retrieval/decision sebelumnya dikosongkan ketika analyze berjalan ulang. `extensions.aurora_contract.run_inputs[run_id]` memuat parent IDs dan hash snapshot. Semua run tetap pada mode bundle.

Sidecar `extensions.aurora_visual.artifacts` menunjukkan path relatif dan SHA-256. ZIP mengangkut coupling/cost/features bersama media; layanan retrieval tidak wajib memproses tensor tersebut. Layout ZIP: bundle.json, media/asset_<sha>.<ext>, artifacts/<run_id>/{transport.json,features.npz}. URI bundle dalam ZIP adalah relatif. Upload ulang raw media via `/api/v1/media` mempertahankan asset_id dan SHA, kemudian ganti URI transport yang berlaku di layanan penerima. Jangan mengirim path filesystem absolut.

Koreksi atom menghasilkan aset atom baru tanpa claim_revision bertambah; perubahan caption/gambar menaikkan claim_revision. Consumer harus menolak referensi atom lama. `probabilities=null` bukan nol dan tidak boleh diganti skor alignment. `Unobservable` tidak otomatis menjadi `InsufficientEvidence` faktual; bukti eksternal dapat mengubah penilaian faktual. Assessment absen/kosong berarti belum dianalisis, bukan U.

Pemeriksaan tim: import ZIP ke instance modul 2 dan 3; validasi schema, hash original image dan hash atom; pertahankan mode; konsumsi nullable probabilities; tolak versi asing dan referensi stale; simpan parent_run_ids. Pada workspace ini kedua repo lain tidak tersedia. Validasi independen di kedua repo belum diklaim lulus.

Laporan verifikasi aktual: `IMPLEMENTATION_STATUS.md` dan `artifacts/reports`. Start/stop lokal tidak menghapus kasus. Untuk melanjutkan model penelitian, bekukan corpus berlisensi dan atom/region gold manusia terlebih dahulu, lalu cache fitur OpenCLIP dan jalankan lima seed/ablation sesuai `configs/research.json`.
