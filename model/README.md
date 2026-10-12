# model: data, pelatihan, evaluasi, distilasi

Skrip Python untuk data Zambia CIDRZ dan TBscreen, pelatihan, evaluasi, dan ekspor bobot int8 ke golden. Hasil dan batasan
ada di `docs/vault/model.md`; keputusan di `docs/vault/keputusan.md`. Data, fitur, bobot, dan laporan disimpan di
`$SUARATB_DATA` (`model/out` adalah symlink ke `runs/`) dan tidak di-commit.

| Subpaket | Isi |
|---|---|
| `data/` | Indeks dataset, pemuatan audio, segmentasi batuk (detektor golden), fitur, audit cakupan data |
| `scratch/` | CNN dilatih dari nol (float, MIL, QAT) dan ekspor int8 ke `QModel` golden |
| `clinical_models/` | Model gejala klinis, fusi dengan skor audio, evaluasi pada semua peserta |
| `evaluation/` | Protokol evaluasi tetap, uji permutasi, lintas-perangkat/dataset/klinik, jalan pintas |
| `pretrained/` | Embedding VGGish, HeAR, OPERA dan evaluasinya |
| `distillation/` | Distilasi HeAR ke CNN golden, kuantisasi dan ekspor bobot |

Prinsip: split selalu per peserta; kolom hasil lab dan prediksi AI X-ray tidak dipakai sebagai fitur; semua metrik dilaporkan
dengan CI 95% bootstrap atas peserta dan nol permutasi bila relevan.
