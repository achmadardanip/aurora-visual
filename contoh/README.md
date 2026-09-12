# Contoh masukan AURORA v1.0.0

Semua nama peristiwa, tempat, dokumen, dan konten pada folder ini **sintetis**. mode=demo wajib dipertahankan. File adalah masukan untuk implementasi mendatang, bukan rekaman layanan yang telah berjalan.

`01_INPUT_RETRIEVAL.json` memiliki analysis=null, retrieval=null, decision=null, dan image=null. Orang 2 dapat langsung membangun query dari caption; keluaran retrieval memakai atom_set_id=null dan atom_ids=[] sampai ada proses remapping yang eksplisit.

`02_INPUT_FUSION_DEMO.json` berisi dua atom yang dianotasi secara eksplisit, Analysis hasil impor tanpa prediksi visual, dua salinan bukti pendukung lokasi dalam satu independence group, dan satu bukti waktu. Kalimat sumber sengaja sederhana untuk menguji baseline verifier; dukungan pada contoh tidak mengukur pemahaman bahasa pada dunia nyata. Atom dan bukti sudah memakai ID/hash sesuai kontrak.

Perilaku yang perlu diperiksa pada file fusi:

1. Importer memvalidasi hash, spans Unicode, atom set, dan referensi bukti.
2. Tidak ada visual_assessments berarti visual belum tersedia; jangan menambahkan prediksi visual Supported atau Unobservable secara diam-diam.
3. Dua salinan bukti lokasi hanya dihitung sebagai satu kelompok sumber, dengan URL null yang sah untuk corpus sintetis lokal.
4. Verifier dapat menelusuri kutipan yang cocok untuk lokasi/waktu. Hasil faktual tergantung implementasi verifier/fusion yang benar-benar dijalankan.
5. Contoh tidak menyediakan artefak kalibrasi. Jika aplikasi belum memiliki artefak demo yang sah dan cocok, confidence_set=null dan abstention_flag=true; keputusan operasional InsufficientEvidence tetap mempertahankan base_label untuk audit.
6. Sinyal detector teks berstatus unsupported karena caption pendek pada provider contoh. Skor null bukan skor nol dan bukan bukti teks ditulis manusia; detector ini bukan layanan nyata.

created_at/retrieved_at adalah timestamp tetap untuk fixture. Image null berarti tidak ada berkas media yang hilang. published_at/first_seen_at/captured_at null berarti waktu sumber tidak diketahui. input.as_of=null berarti fixture tidak meminta keputusan historis dengan cutoff ketat.

Saat implementasi, tambahkan golden vectors dengan Unicode, emoji, float, serta key order berbeda dan jalankan validator yang dibangun dari kontrak. Contoh paket ini sengaja menggunakan nilai numerik yang serialisasinya sederhana; pemeriksaan hash-nya tidak menggantikan suite kesesuaian penuh RFC 8785.
