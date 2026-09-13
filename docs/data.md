# Data, lisensi dan anotasi

Sumber primer diperiksa pada 12 September 2026. Salinan README yang dibaca dan hasil pemeriksaan lisensi tersedia di `artifacts/reports/*-source.txt` dan `source-verification.json`. Tidak ada dataset besar/bobot berlisensi yang diunduh otomatis.

**MAFINDO V1 bukan dataset training visual AURORA.** Uji live hanya mengembalikan metadata fact-check yang dinormalisasi (ID, judul, klasifikasi/status, tanggal), tanpa pasangan byte gambar–caption, region gold, label atom S/C/U, atau grant lisensi training/redistribusi yang telah diverifikasi. Karena itu respons API tidak diimpor sebagai corpus penelitian dan tidak dikirim ke GPU. Penggunaan data untuk training baru boleh dilakukan setelah pemilik data memberikan ketentuan lisensi tertulis, media diperoleh secara sah, dan anotasi visual sesuai kontrak tersedia.

| Sumber primer | Format aktual yang diperiksa | Label dan akses |
|---|---|---|
| [NewsCLIPpings](https://github.com/g-luo/news_clippings) (master README) | JSON `annotations`: id, image_id, falsified; join ke VisualNews `origin/data.json` untuk caption/image_path | Label pasangan pristine/falsified, bukan gold atom. GitHub tidak menyajikan LICENSE pada endpoint lisensi yang diperiksa; hak VisualNews/media harus diperiksa sebelum penggunaan/redistribusi. |
| [VERITE](https://github.com/stevejpapad/image-text-verification) (master README) | `VERITE.csv`: caption, image_path, label | true/miscaptioned/out-of-context, tingkat pasangan. Repo Apache-2.0; media pihak ketiga bukan otomatis di bawah lisensi kode. README pembaruan Juli 2026 mengarahkan raw images ke HF dengan verifikasi email institusi; belum diakses. |
| [COSMOS](https://github.com/shivangi-aneja/COSMOS) (README.MD) | JSON train/val: img_local_path, articles[].caption; test: caption1, caption2, label, maskrcnn_bboxes | Training tanpa anotasi out-of-context; test ooc/not-ooc berlaku pada konteks pasangan caption. Akses dataset melalui formulir. Repo MIT; ketentuan media tetap ditinjau. |
| [MMFakeBench](https://github.com/liuxuannan/MMFakeBench) (main README) | JSON text, image_path, text_source, image_source, gt_answers, fake_cls | Label real/fake dan tipe distorsi tingkat pasangan. README menyatakan CC-BY 4.0 dan mewajibkan Data Usage Protocol di Hugging Face. Tidak menyamakan fake dengan kontradiksi visual. |

Importer CLI menyediakan adapter format di atas untuk berkas lokal yang diperoleh secara sah. Pengguna memberikan image-root, split, license-note, serta groups.json; tidak ada download URL input. Pada COSMOS caption diperluas menjadi baris terpisah, pair_context_label disimpan sebagai metadata dan post_label null. Proposal box sumber tidak diangkat menjadi gold grounding.

```sh
uv run aurora import-dataset --source verite --input /data/VERITE.csv \
  --image-root /data/VERITE --groups /data/groups.json --split test \
  --license-note 'Research access approved; original media rights retained' \
  --output data/manifests/verite.jsonl
```

`groups.json` memetakan indeks baris rilis asli ke `{event_id,source_group,parent_id,source_url}`. Untuk NewsCLIPpings tambahkan `--visualnews /data/visualnews/origin/data.json`. Semua variasi satu gambar/peristiwa/sumber/parent harus satu split; hashes gambar identik tidak boleh melintas split. Diperlukan peninjauan event/source oleh manusia, karena hash piksel tidak menemukan near-duplicate atau sindikasi otomatis.

## Manifest internal

JSONL divalidasi `Example` di `training/data.py`. Field wajib: sample_id, image (relatif manifest), image_sha256, caption persis, language, dataset, split, event_id, source_group, parent_id, license, source_url, post_label, annotations, data_kind dan extensions (default {}). `annotations` berisi `{atom: Atom lengkap, label: S/C/U atau null, regions: normalized bbox[], provenance: human/weak/fixture/unreviewed, quality: skor terukur atau null}`. `data_kind` research/fixture wajib eksplisit. Unreviewed tidak boleh memasok gold label. Training menolak atom tanpa label; cache-features tetap dapat memproses data tanpa anotasi.

`extensions.hard_positive_id` menunjuk sample dengan image dan split sama; `atom_correspondence` adalah array pemetaan indeks atom asal ke indeks atom pasangan, bijektif dan dengan gold label sama. Grid kolomnya identik karena gambar/config sama. Generator weak supervision mengeluarkan role/entity/attribute/relation swap, number change, negation serta synonym/voice/paraphrase. Ukuran edit dan kelulusan aturan grammar dicatat; confidence empiris null. Perubahan tanpa observasi alternatif tidak menjadi C. Label hard positive hanya diwariskan jika label sumber diberikan; perubahan makna kecil tidak membuktikan observabilitas.

## Benchmark atomisasi manusia

`data/benchmarks/pilot-candidates.jsonl` terpisah dari fixture unit test. Isinya kandidat caption ID/EN dengan kondisi negasi, peran, angka, tanggal, lokasi, sebab, koreferensi, Unicode dan voice. `gold_atoms` serta identitas/review dua anotator masih null: **belum merupakan benchmark ditinjau manusia**. Kode tidak mengarang review manusia.

Pilot 100 dan target 500–1.000 contoh merupakan rencana riset. Dua anotator menyusun atom minimal, spans, dependensi dan label/region independen; adjudikasi mencatat perbedaan dan keputusan. Bekukan split berdasarkan event/image/source sebelum tuning. Jalankan evaluasi parser hanya setelah gold_atoms dan review tersedia. Jangan gunakan contoh UI/fixture pada evaluasi confirmatory.

## Provenance dan screening asal media

Setiap upload dapat menghasilkan `extensions.aurora_visual.screening` pada run analysis. Screening membaca byte asli dan hanya mempertahankan nama field metadata/presence indicator, bukan nilai EXIF mentah yang mungkin sensitif. Marker perangkat lunak generatif yang dikenal dapat menghasilkan `likely_ai_generated`; metadata kamera tanpa marker kuat dapat menghasilkan `no_strong_ai_signal`; sisanya `inconclusive`. Semua label ini terbatas pada pemeriksaan lokal dan selalu disertai batas bahwa mereka tidak memengaruhi S/C/U maupun kebenaran caption.

Pencarian `c2pa`, `jumbf`, atau `content credentials` hanya mendeteksi marker byte. Tidak ada penguraian manifest, verifikasi tanda tangan, trust store, atau pembuktian rantai provenance. Watermark tak terlihat, AI-image, dan deepfake tidak dinilai sampai model tervalidasi dikonfigurasi. Metadata dapat dihapus atau dimodifikasi; ketiadaan sinyal tidak membuktikan gambar berasal dari kamera.


[OpenCLIP](https://github.com/mlfoundations/open_clip) diperiksa dari README primer; encoder pretrained harus dipilih beserta lisensi/model card datasetnya. Dukungan Indonesia tidak diasumsikan dari kemampuan tokenizer menerima teks. FG-CLIP/SAM2/TextRegion belum diaktifkan karena checkpoint/lisensi dan dependensi tidak diverifikasi. [Tesseract tessdata](https://github.com/tesseract-ocr/tessdata) menjelaskan traineddata; engine lokal menyediakan ind dan eng dan telah diuji. OCR score adalah skor engine, bukan confidence faktual.
