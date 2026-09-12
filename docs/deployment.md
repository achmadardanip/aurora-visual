# Deployment satu mesin/server

Penggunaan lokal: `make setup && make dev`; port 8101/5171 default hanya loopback. Environment selalu server-side untuk checkpoint dan API key. `make dev` menolak bind non-loopback tanpa `AURORA_PUBLIC=true`; proses API menolak konfigurasi public dengan token kurang dari 24 karakter.

Compose menjalankan dua service mandiri API/frontend dengan volume `aurora-data`. API mempunyai worker supervisor bawaan; jangan menambah uvicorn workers secara sembarang. Nginx menyajikan build statis dan proxy API, request body max 40MB, rate limit 10 request/detik burst 20. API membatasi upload 10MB, 25 megapixel, antrean 20 per pemilik dan job timeout 180 detik. Parameter dapat diubah via environment; sesuaikan dengan kapasitas.

Sebelum membuka ke jaringan luar: atur `AURORA_PUBLIC=true`, token acak kuat, `AURORA_CORS` hanya origin frontend sebenarnya, dan reverse proxy TLS seperti `configs/Caddyfile.example`. UI meminta token lewat halaman Metode dan menyimpannya hanya selama sesi browser. API mengecek token untuk resolve media dan history, bukan hanya submit. Jangan memasukkan token ke query URL atau VITE env. Mode lokal terbuka tidak dianggap siap produksi. Caddy/domain deployment memerlukan konfigurasi administrator dan belum dijalankan.

Hentikan aplikasi sebelum backup sederhana seluruh `var/` (atau gunakan SQLite backup API saat aktif), termasuk media/derived/artifacts bersama database. Jangan hanya menyalin database tanpa sidecar. Restore dengan app berhenti, pertahankan permission folder, lalu jalankan Alembic dan `/ready`. Rekam versi app/lockfile bersama backup. Migrasi destruktif downgrade sengaja tidak tersedia.

Tidak ada URL fetch dari input pengguna. Origin LLM yang di-allowlist adalah endpoint internal/admin yang tepercaya; backend menolak redirect. Jangan memasukkan endpoint yang dapat dikendalikan publik ke allowlist internal. Escape teks/OCR melalui rendering React biasa; tidak ada dangerouslySetInnerHTML. ZIP diinspeksi sebelum penulisan dan tidak memakai extractall. CSP frontend produksi membatasi sumber ke self/blob, script self, frame-ancestors none; style inline dibutuhkan posisi region.

Perluasan multi-user/produksi besar memerlukan IdP, rotasi token, audit otorisasi, rate limiting per pengguna, PostgreSQL/worker lease terdistribusi, artifact storage privat dan load testing. Dokumen ini tidak mengklaim pengujian produksi telah selesai.
