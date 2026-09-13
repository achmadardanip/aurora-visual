# Deployment satu mesin/server

Penggunaan lokal: `make setup && make dev`; port 8101/5171 default hanya loopback. Environment selalu server-side untuk checkpoint dan API key. `make dev` menolak bind non-loopback tanpa `AURORA_PUBLIC=true`.

## Profil produksi yang didukung

Target operasional yang didukung saat ini adalah **satu pemilik pada satu mesin**: satu API/worker, SQLite WAL, media dan sidecar pada volume privat, frontend Nginx, serta TLS termination di Caddy/reverse proxy. Ini dapat dipakai untuk layanan privat bertrafik terbatas setelah checklist di bawah dipenuhi. Ini bukan arsitektur SaaS multi-tenant atau cluster horizontal.

Konfigurasi wajib untuk ekspos publik:

```dotenv
AURORA_PUBLIC=true
AURORA_API_TOKEN=<minimal-32-random-characters>
AURORA_CORS=https://aurora.example.org
AURORA_ALLOWED_HOSTS=aurora.example.org
```

Wildcard origin/host ditolak. Public mode juga menolak host localhost dan menonaktifkan endpoint diagnostik MAFINDO; key MAFINDO tidak perlu berada pada instance publik. API menambahkan HSTS, no-referrer, nosniff, permissions policy, dan frame denial. TLS tetap wajib di reverse proxy.

## Menjalankan container

```sh
docker compose up --build -d
```

Compose menjalankan API dan frontend dengan volume `aurora-data`; port host tetap loopback agar hanya reverse proxy yang mengeksposnya. API mempunyai worker supervisor bawaan—jangan menambah uvicorn worker atau replica API pada SQLite. Startup container menjalankan migrasi Alembic sebelum Uvicorn. Nginx membatasi request dan meneruskan timeout analisis 210 detik. Docker image API berjalan sebagai UID/GID 10001 non-root, menyertakan CLI backup/verify/restore, dan versi base image dipatok.

Build serta smoke lokal lulus pada 13 September 2026 dengan Docker Desktop 29.7.2, Compose 5.5.1, dan host arm64. Pemeriksaan mencakup migrasi `0001`, health/readiness langsung dan melalui Nginx, syntax Nginx, UID/GID dan write access volume, browser Tahap 0–3 dari origin frontend container tanpa error console/network, persistensi kasus/media dan replay idempotency setelah restart, public-mode token/Host/CORS/HSTS serta MAFINDO 404, dan backup/verify/restore volume bernama saat layanan dihentikan. Tidak ada `.env`, `.claude`, atau `prompt-aurora` dalam filesystem image yang diperiksa, dan Uvicorn tidak mengirim header `Server`. Nginx tetap mengirim banner pada hop loopback; contoh Caddy menghapusnya pada respons eksternal, tetapi Caddy/TLS belum diuji runtime. Docker Scout tersedia tetapi pemindaian CVE belum dijalankan karena memerlukan login Docker.

Gunakan `configs/Caddyfile.example` setelah mengganti domain. Pastikan DNS, sertifikat, firewall, permission volume, dan retensi log dikonfigurasi administrator. Jangan menaruh token di query URL, image frontend, `VITE_*`, repo, atau log.

## Backup dan restore drill

Backup online konsisten memakai SQLite backup API, menyalin file terkait ke staging privat, lalu membuat dan memverifikasi archive ber-manifest SHA-256 sebelum publikasi atomik. Jalankan pada periode tanpa upload/import/analisis aktif atau hentikan aplikasi terlebih dahulu: tiap file di-snapshot secara konsisten dan diverifikasi, tetapi transaksi yang berjalan serentak dapat menghasilkan set database/file dari momen berbeda.

```sh
make backup
```

Verifikasi archive sebelum menyalin atau memulihkan:

```sh
make verify-backup ARCHIVE=backups/aurora-backup-YYYYMMDDTHHMMSSZ.tar.gz
```

Restore hanya ke directory baru/kosong dan saat aplikasi berhenti:

```sh
make restore-backup ARCHIVE=backups/aurora-backup-YYYYMMDDTHHMMSS.ffffffZ.tar.gz RESTORE_DIR=/path/kosong
```

Perintah restore memvalidasi schema manifest, jumlah/jenis/path/ukuran member, checksum, batas ukuran, dan integritas SQLite tanpa mengekstrak path archive secara langsung. Setelah berhasil, pertahankan permission owner service, arahkan salinan konfigurasi ke directory hasil restore, jalankan Alembic, lalu periksa `/ready`. Simpan archive terenkripsi di lokasi terpisah dan jalankan drill berkala pada directory disposable. Jangan arahkan output backup ke dalam `AURORA_DATA_DIR`; perintah akan menolaknya.

## Batas dan operasi

Batas default: upload 10 MB/25 megapixel, antrean 20 per pemilik, satu job komputasi, timeout 180 detik. Monitor disk volume, error/latency reverse proxy, status `/ready`, heartbeat worker, pertumbuhan audit/snapshot, umur backup, dan kegagalan job. Rotasi token membutuhkan restart layanan dan sesi browser baru. Identitas pemilik saat ini merupakan hash token, sehingga mengganti token membuat kasus/media lama tidak terlihat melalui API; rencanakan ekspor/migrasi atau pertahankan token yang sama selama masa retensi.

Tidak ada URL fetch dari input pengguna. Origin LLM allowlist hanya untuk endpoint internal tepercaya; redirect ditolak. React tidak menggunakan HTML mentah. Import ZIP memeriksa traversal, symlink, duplikat, bomb, dan hash sebelum penulisan.

## Belum didukung

Produksi multi-user/berskala besar masih memerlukan IdP/SSO, registry izin, rotasi credential tanpa restart, PostgreSQL, klaim job terdistribusi, object storage privat, observability terpusat, load/chaos testing, dan runbook insiden. Model penelitian umum juga belum tersedia: MAFINDO bukan corpus visual berlisensi, dan GPU tidak mengatasi ketiadaan pasangan gambar–caption berlisensi beserta anotasi atom/region. Jangan mengklaim kemampuan tersebut sampai data, lisensi, model, dan evaluasinya tersedia.
