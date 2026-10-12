# Indeks konteks SuaraTB

Baca berurutan: `vault/status-fase.md` -> `vault/spesifikasi.md` -> `vault/keputusan.md`.

| Berkas | Isi |
|---|---|
| `vault/status-fase.md` | Fase aktif, hal terbuka |
| `vault/spesifikasi.md` | Parameter final (ringkasan dari `golden/params.py`) |
| `vault/keputusan.md` | Keputusan K-01..K-23 beserta alasan dan angka |
| `vault/kalibrasi-fase0.md` | Tabel SNR/galat/detektor (sintetis) |
| `vault/data.md` | Dataset: sumber, lisensi (ODbL-1.0 dari metadata Kaggle), struktur, label |
| `vault/kalibrasi-data-nyata.md` | Kalibrasi pada audio nyata (FFT_W, FEAT_OFFSET, detektor) |
| `vault/model.md` | SATU dokumen model: arsitektur int8, data, pelatihan, kuantisasi, capaian, batasan (hasil negatif) |
| `vault/verifikasi.md` | Infrastruktur verifikasi Fase 1 |
| `vault/sintesis.md` | Hasil sintesis Vivado XC7A35T-1 dan Yosys: sumber daya, timing, sapuan P, versi 150 MHz |
| `result/result.md` | Laporan hasil lengkap (dan `result/result.pdf`) |
| `vault/anggaran-fase0.md` | Keluaran `golden/budget.py`: memori, pengali, siklus, muat/tidak per perangkat |

## Kode
| Jalur | Fungsi |
|---|---|
| `golden/params.py` | Sumber tunggal parameter, deskriptor layer, paket SV |
| `golden/blocks/` | preemph, window, fft512, power, mel, log_q, quant, cnn, cough_det, frontend (+fixed) |
| `golden/gen_tables.py` | ROM .mem + paket SV -> `golden/tables/` |
| `golden/gen_vectors.py` | Vektor uji sintetis per blok -> `tb/vectors/` (CNN: bobot ACAK) |
| `golden/budget.py` | Anggaran dan penilaian muat |
| `golden/calibrate.py` | Eksperimen Fase 0 (`--data data/raw` untuk audio nyata) |
| `golden/synth.py` | Audio sintetis (bukan klinis) |
| `golden/test_*.py` | 61 tes pytest |

Pemetaan golden -> RTL (Fase 2-5): preemph->`preemph.sv`, window->`window.sv`, fft512->`fft512.sv`, power->`power.sv`,
mel->`mel.sv`, log_q->`log_q.sv`, quant->`quant.sv`, cough_det->`cough_det.sv`, cnn->`pe_array/cnn_ctrl/requant/gap_fc`.

## Kode tambahan (Fase M, 1, 2)
| Jalur | Fungsi |
|---|---|
| `model/{dataset,audio_io,prep_audio,build_data}.py` | Indeks data, muat WAV 16 kHz, segmentasi dengan detektor golden, fitur + split per peserta |
| `model/{mil_train,fusion,export}.py` | Pelatihan MIL, fusi gejala, PTQ/QAT -> QModel golden |
| `golden/calibrate_real.py` | Kalibrasi pada klip nyata |
| `rtl/common/*`, `rtl/{audio,detect}/*`, `rtl/files.f` | Primitif umum + front-end Fase 2 |
| `tb/common/*`, `tb/test_*.py`, `Makefile` | Infrastruktur verifikasi (dijalankan di Fase T) |
