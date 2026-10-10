# Indeks konteks SuaraTB

Baca berurutan: `vault/status-fase.md` -> `vault/spesifikasi.md` -> `vault/keputusan.md`.

| Berkas | Isi |
|---|---|
| `vault/status-fase.md` | Fase aktif, hal terbuka |
| `vault/spesifikasi.md` | Parameter final (ringkasan dari `golden/params.py`) |
| `vault/keputusan.md` | Keputusan K-01..K-12 beserta alasan dan angka |
| `vault/kalibrasi-fase0.md` | Tabel SNR/galat/detektor (sintetis) |
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
