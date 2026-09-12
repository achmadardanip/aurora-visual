# Training dan evaluasi

`make evaluate` menjalankan training CPU fixture lalu evaluasi kecil. Fixture memakai tensor sintetis dengan pasangan hard-positive yang korespondensinya diketahui. Tujuannya memeriksa gradien, loss finite, penyimpanan optimizer/RNG/config, load/resume/predict, coupling dan metrik. Jangan melaporkan hasilnya sebagai akurasi dataset publik.

Training mempelajari text/region projector, role projector, bobot cost dan 3-state head; backbone beku. Loss default: class-balanced CE (1), contrastive atom–region (0.1), support/contradiction transport margin (0.1), hard-positive symmetric KL + coupling MSE (0.2), negative entropy (0.001). Margin hanya dihitung bila sampel memuat pasangan S/C; target region hanya untuk supported atom dengan anotasi region. Entropy, convergence, finite gradients dan validation CE dicatat. Weights/tuning memakai validation, bukan test atau calibration. CLI saat ini memakai learning rate/config eksplisit; tidak otomatis mengklaim hyperparameter optimal.

Checkpoint `.pt` memuat format version, state model/optimizer, RNG PyTorch, config, seed, backbone_version, label map bernama, embedding dim, data/split hashes dan IDs/groups, steps/epoch/history. Sidecar JSON memuat hash checkpoint; loader memeriksa hash dan kesesuaian metadata. `torch.load(weights_only=True)` tidak melakukan unpickle arbitrer. Ekspor safetensors untuk handoff bobot, metadata tetap terpisah. Smoke checkpoint `data_kind=fixture` dilarang dalam inferensi live atau evaluasi confirmatory.

CLI `predict` dan `evaluate` memeriksa kesamaan identitas fitur terhadap checkpoint sebelum inferensi, termasuk encoder, versi bobot, preprocessing, dan grid. Metode alignment default mengikuti metadata checkpoint; `--method` hanya digunakan untuk penggantian eksplisit. Ini mencegah checkpoint baseline lain dievaluasi diam-diam dengan UOT. Metode yang dijalankan tercatat dalam laporan.

## Baseline dan ablation

`make experiments` melatih ulang 11 kondisi pada dataset/split/seed sama: global image–full-caption (kehilangan pembeda atom), max region, mean region, attention, balanced OT, UOT, cosine-only learned cost, tanpa unmatched, tanpa hard positives, text-only dan image-only. Baseline global memakai full-caption embedding pada data nyata; smoke hanya memiliki global tensor sintetis. Probe modality mematikan fitur modality lain beserta cue turunan. Ini eksperimen engineering kecil, bukan pembuktian bias model.

`configs/research.json` menetapkan lima seed (17,29,43,71,101), split train/validation/calibration/test, semua metode, robustness dan unit bootstrap event. Bandingkan parser rules/LLM pada gold benchmark yang sama setelah parser opsional tersedia. Parser opsional yang belum dikonfigurasi dilaporkan unavailable, bukan angka nol atau hasil palsu. Evaluasi lintas dataset mengelompokkan per-dataset metrics; dataset publik belum tersedia saat implementasi.

## Metrik

- Atom extraction: one-to-one Hungarian matching; role dan SPO/negasi/kuantitas persis serta span code-point IoU ≥0.5. Precision/recall/F1 ini ketat; sinonim perlu gold adjudication, tidak otomatis dianggap cocok.
- Atomic macro-F1 S/C/U, per-label dan contradiction F1, confusion per role, accuracy, false-contradiction rate dengan denominator gold non-C. Label yang absen diberi F1=0 dalam macro tiga kelas; denominator nol pada rate khusus menjadi null.
- Role-swap accuracy hanya pada pasangan berlabel `role_swap`; null bila tidak tersedia. Hard-positive consistency mengikuti mapping baris eksplisit. Group accuracy adalah seluruh prediksi benar pada setiap event.
- IoU dan pointing game hanya bila gold bbox tersedia. Nilai null jika tanpa gold, bukan nol.
- Deletion/insertion AUC, sufficiency gap, comprehensiveness memakai intervensi **feature space**: region embedding dihapus/ditambahkan dengan zero baseline dan global context dihitung ulang. Nilai bukan skor alignment dan bukan bukti faithfulness kausal piksel. Kurva lengkap dicatat.
- Bootstrap 95% CI resample event, bukan atom. Kurang dari dua event → CI null beserta alasan. Lima seed dikonfigurasi; smoke satu seed tidak cukup untuk klaim statistik.
- Waktu parsing/inferensi, peak RSS proses (byte di macOS, KiB Linux), CUDA peak bila tersedia. Target GPU 16–24GB belum diprofilkan. Pengukuran peak proses tidak sama dengan memori incremental model.

`robustness.py` memberi perturbasi piksel blur, crop (dengan transform bbox), dan overlay OCR. Evaluasi perubahan tersebut harus memeriksa bahwa crop mengurangi observabilitas; label asli tidak otomatis dipertahankan ketika bukti hilang. Tulisan tanggal overlay tidak menjadi bukti tanggal pengambilan foto. Data human-reviewed belum tersedia untuk metrik robustness penelitian.

`uv run aurora robustness --smoke --checkpoint artifacts/checkpoints/smoke.pt --output artifacts/reports/robustness-smoke.json` memeriksa jalur ekstraksi fitur dari piksel dan empat variasi gambar. Checkpoint smoke dilatih pada tensor sintetis, sehingga laporan menandai perbedaan fitur sebagai `fixture_override_unvalidated`. Nilai kesepakatan prediksi hanya hasil pengujian wiring. Jalur manifest/checkpoint penelitian menolak ketidakcocokan identitas fitur dan mengikuti metode alignment checkpoint.

Contoh jalur penelitian setelah data/anotasi diperoleh:

```sh
uv run aurora cache-features --manifest data/manifests/reviewed.jsonl --backbone openclip
uv run aurora train --manifest data/manifests/reviewed.jsonl --backbone openclip --epochs 20 --output artifacts/checkpoints/research.pt
uv run aurora evaluate --manifest data/manifests/reviewed.jsonl --checkpoint artifacts/checkpoints/research.pt --backbone openclip --confirmatory
```

Contoh perintah bukan bukti sudah dieksekusi. Manifest dievaluasi terhadap overlap image/event/source, checkpoint smoke ditolak, dan confirmatory memerlukan gold atom dengan provenance human. Kalibrasi conformal utama, pengambilan evidence eksternal dan verdict faktual tidak diimplementasikan oleh modul ini.
