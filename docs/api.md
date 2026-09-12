# API AURORA Visual 1.0.0

Default `http://127.0.0.1:8101`; schema di `/api/v1/schema`, dokumentasi interaktif `/docs`. Semua endpoint proses mengikuti [kontrak normatif](../prompt-aurora/KONTRAK_BERSAMA.md). Definisi lengkap: `contracts/aurora.ts`, `contracts/aurora.schema.json`, `contracts/openapi.json`, dan `backend/app/models/contract.py`.

| Endpoint | Fungsi |
|---|---|
| GET `/health` | Liveness + service + schema_version |
| GET `/ready` | Database, heartbeat worker, capabilities; 503 bila dependensi lokal gagal, 200 degraded untuk provider opsional |
| GET `/api/v1/providers/mafindo/latest?limit=1` | Diagnostik kompatibilitas MAFINDO baca-saja; hanya server-side, tidak menambah kasus/bukti/verdict |
| POST `/api/v1/media` | Multipart field `image`; decode/MIME/size/hash/EXIF; mengembalikan MediaRef |
| GET `/api/v1/media/{asset_id}` | Byte asli yang diizinkan untuk pemilik; `?preview=true` untuk RGB oriented |
| POST `/api/v1/analyze` | Body AuroraBundle, header Idempotency-Key; 202 job envelope |
| GET `/api/v1/jobs/{job_id}` | `{job_id,status,result,error,progress,attempts}` |
| POST `/api/v1/jobs/{job_id}/cancel` | Menghentikan queued/running; terminal failed + CANCELLED |
| GET `/api/v1/jobs/{job_id}/bundle` | Unduh JSON hasil job, termasuk snapshot lama |
| GET `/api/v1/cases?limit=50` | Daftar terbaru, batas 100 |
| GET `/api/v1/cases/{case_id}` | Bundle terkini |
| GET `/api/v1/cases/{case_id}/history` | Snapshot revisi, alasan, parent, bundle |
| PATCH `/api/v1/cases/{case_id}/atoms` | Koreksi eksplisit dengan kontrol stale |
| GET `/api/v1/cases/{case_id}/export?format=json\|zip\|csv\|overlay` | Ekspor; overlay dapat difilter `atom_id` |
| POST `/api/v1/import` | Multipart field `file`, JSON/ZIP; respons `{bundle,asset_available}` |
| POST `/api/v1/demo/{supported\|contradicted\|unobservable}` | Menghasilkan input fixture dan media; belum menganalisis |

## Screening asal media dan MAFINDO

Hasil analysis dapat memuat `extensions.aurora_visual.screening`. Objek lokal ini berisi target media, field metadata yang hadir, klasifikasi marker perangkat lunak, marker C2PA/JUMBF, status detector, dan keterbatasan. Ia tidak mengubah kontrak top-level, `analysis.visual_assessments`, `retrieval`, atau `decision`. `marker_present` berarti byte marker ditemukan saja; `verification=not_verified` bukan validasi provenance kriptografis. `unavailable` pada detector berarti tidak dinilai, bukan hasil negatif.

`GET /api/v1/providers/mafindo/latest` membaca key dari konfigurasi server (`AURORA_MAFINDO_API_KEY`) dan hanya memanggil provider ketika key tersedia. Parameter `limit` dibatasi 1–20. Response dinormalisasi untuk diagnosis dan tidak ditulis ke database kasus; klasifikasi maupun status provider tidak boleh dipakai sebagai label visual atau verdict faktual. Error yang dikembalikan sengaja generik dan tidak memuat URL/key.


1. Upload gambar, gunakan MediaRef yang dikembalikan.
2. Buat AuroraBundle: case_id UUID lowercase, claim_revision=1, mode demo/live, created_at RFC3339, input caption asli/bahasa/media/as_of, analysis/retrieval/decision null, warnings=[], extensions={}.
3. POST analyze memakai Idempotency-Key unik sepanjang 1–128 karakter. Poll job sampai terminal. Key dan payload sama (JCS request sebelum modifikasi) mengembalikan job yang sama; perubahan URI juga berarti payload berbeda dan menghasilkan 409.
4. Key scoped pada pemilik+endpoint, disimpan tanpa expiry otomatis dalam database lokal. Jangan menghapus key untuk me-retry request jaringan; untuk komputasi baru gunakan key baru. Retensi sama dengan database; penghapusan manual hanya melalui kebijakan backup/retensi administrator.

Contoh opsi internal:

```json
{"aurora_visual":{"options":{"backbone":"local-color-v1","alignment":"uot","parser":"rules","head":"heuristic","top_k":16}}}
```

Backbone: local-color-v1/openclip. Alignment: global/max-region/mean-region/attention/balanced-ot/uot. Parser: rules/llm. Head: heuristic/trained. Checkpoint/path/URL model tidak menerima input bebas dari browser; dikendalikan environment server. Head trained membutuhkan metadata split, label map bernama, hash checkpoint cocok, dan dataset_kind research. Pemilihan model tersedia juga lewat API; UI default memakai heuristic.

Koreksi:

```json
{"expected_atom_set_id":"aset_<hash saat ini>","atomic_claims":["objek Atom lengkap, bukan string ini"],"reason":"Alasan koreksi"}
```

Atom ID yang diimpor dipertahankan. Spans Unicode code point, end eksklusif; UI memotong `Array.from(caption)`, bukan unit UTF-16. Referensi dependen tidak boleh dangling/cyclic. Atom set dihitung ulang; request dengan set lama 409. Untuk caption/gambar baru, kirim revisi+1 serta ketiga output null. Mode tidak diubah dalam kasus yang sama; buat kasus baru.

Error: `{error:{code,message,retryable,details}}`. Validasi/SCHEMA_VERSION_UNSUPPORTED 422, revisi/atom/key konflik 409, media terlalu besar 413, antrean penuh 429, objek tidak ditemukan 404. Kegagalan job tampil terminal failed dengan result=null, bukan hasil berhasil. Partial result memuat peringatan nyata. Unobservable adalah status semantik; job gagal tidak disamarkan sebagai U.

Mode publik mewajibkan `Authorization: Bearer <token>` untuk seluruh API/media; `/health` tetap tidak sensitif. Token disimpan di sesi browser melalui halaman Metode, tidak dibundel ke frontend. Satu token = satu pemilik; autentikasi multi-user/SSO belum dibuat. API tidak mengambil URL media eksternal: aset harus di-upload/relay, menutup jalur SSRF input. URL parser opsional hanya berasal dari konfigurasi server dengan allowlist origin dan tanpa redirect.

JSON metadata tanpa media dapat diimpor; analisis ulang baru diizinkan setelah byte dengan SHA cocok tersedia. ZIP memuat bundle.json, media relatif dan artifacts dengan hash. Importer menolak path absolut, `..`, backslash, symlink, duplikat, file ekstra tanpa referensi, archive terenkripsi, mismatch hash/dimensi, >100 entry, >50MiB uncompressed, >20MiB per file, dan rasio kompresi >1000. Tidak menggunakan extractall dan tidak mengirim path lokal antarlayanan.
