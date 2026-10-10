# Status fase

**Fase aktif: 0 selesai (golden + tes hijau, kalibrasi SINTETIS).** Menunggu instruksi Fase 1.
Diperbarui: 2026-10-10.

| Fase | Status | Catatan |
|---|---|---|
| 0 | Selesai, 61 tes golden lulus | Parameter dibekukan di `golden/params.py`; ⚠ konstanta data-dependen provisional |
| M | Belum | Dataset Kaggle belum diunduh/diverifikasi; bobot terlatih belum ada |
| 1-5 | Belum | Tulis saja, jangan jalankan simulasi |
| T | Belum | |
| 6 | Belum | Vivado belum diperiksa pada sesi ini |

## Sebelum Fase 1 / yang masih terbuka
1. `data/raw` kosong: ulangi `python -m golden.calibrate --data data/raw` untuk FEAT_OFFSET, detektor, dan AUROC
   (syarat "AUROC turun ≤ 0,01" untuk lebar FFT belum diperiksa).
2. Requant/bobot CNN: dari Fase M (QAT); golden hanya punya model acak + kalibrator.
3. Tafsiran segmen 50 frame (K-03) dan jadwal skala FFT `first1_0` (K-04): perlu konfirmasi.
4. Kapasitas perangkat dan jenis FPGA panitia: dari ingatan / belum diumumkan.
5. Mikrofon INMP441 mengeluarkan 24 bit; spesifikasi memakai 16 bit teratas. Bila bit bawah dipakai, derau
   kuantisasi masukan turun dan lebar FFT mungkin perlu dihitung ulang (tidak dievaluasi).
6. Makefile belum ada (Fase 1). Perintah: `.venv/bin/python -m pytest golden`, `-m golden.gen_tables`,
   `-m golden.gen_vectors`, `-m golden.budget`, `-m golden.calibrate`.
