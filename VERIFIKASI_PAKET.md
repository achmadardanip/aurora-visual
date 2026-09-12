# Verifikasi paket prompt AURORA

Tanggal: 12 September 2026. Objek pemeriksaan adalah paket prompt dan contoh masukan, bukan aplikasi hasil implementasi.

- Tiga berkas prompt lengkap tersedia, masing-masing dengan lingkup, alur pengguna, metode, pengujian, evaluasi, dan perintah implementasi.
- Bagian kontrak di ketiga prompt identik dengan KONTRAK_BERSAMA.md; versi yang digunakan 1.0.0.
- Seluruh tautan berkas lokal dalam Markdown menunjuk berkas yang tersedia; code fences berpasangan dan teks UTF-8 terbaca.
- Dua contoh JSON dapat diparsing dan berlabel demo; struktur tingkat utama, ID, referensi atom, spans, hash teks/bukti/atom set, serta mode diperiksa.
- Contoh fusi memiliki dua atom, tiga bukti dalam dua kelompok sumber independen, dan satu sinyal detector teks unsupported tanpa skor palsu.
- Ketentuan AI detector gambar/teks yang dapat diganti tercakup pada orang 2; fusi orang 3 memisahkan sinyal tersebut dari keputusan faktual.
- Label visual S/C/U dan label faktual S/C/IE dipisahkan; kalibrasi atom dan klaim, quantile finite-sample, null/empty set, serta abstensi dijelaskan pada prompt orang 3.

Contoh JSON menggunakan nilai numerik sederhana untuk pemeriksaan hash. Kesesuaian lengkap RFC 8785 lintas runtime, validasi schema/model Pydantic, serta tests aplikasi harus dijalankan ketika ketiga proyek diimplementasikan. Tidak ada hasil model, request provider live, eksperimen ilmiah, atau performa aplikasi yang diklaim oleh verifikasi paket ini.
