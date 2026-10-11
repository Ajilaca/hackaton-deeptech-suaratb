# Status fase

**Fase aktif: M, 1, 2 selesai (2026-10-11). Berikutnya Fase 3 (DSP) setelah pengguna meminta; Fase 1-5 hanya menulis kode.**
Diperbarui: 2026-10-10.

| Fase | Status | Catatan |
|---|---|---|
| 0 | Selesai, 61 tes golden lulus | Parameter dibekukan di `golden/params.py`; ⚠ konstanta data-dependen provisional |
| M | Selesai (pipeline), HASIL KLINIS NEGATIF | 360 peserta/58 positif; batuk saja AUROC 0,562 ≈ nol permutasi; int8 golden vs float Δ +0,001; bobot di `model/out/model_v2` (tidak di-commit). Lihat `model.md` |
| 1 | Ditulis (belum dijalankan) | Makefile, rtl/common (rom/ram/fifo/sva), tb/common (vec, cocotb_utils, runner), `docs/vault/verifikasi.md` |
| 2 | Ditulis (belum dijalankan) | `i2s_rx`, `preemph`, `frame_buf`, `cough_det` + 4 testbench cocotb + vektor golden |
| 3-5 | Belum | Tulis saja, jangan jalankan simulasi |
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

## Pembaruan data nyata (2026-10-11)
- `data/raw` -> `/data/suaratb/raw`; perekam (281 WAV) selesai, Phone A (3 klinik, 26,3 GB) masih diunduh.
- Segmentasi perekam: 280 rekaman -> 3.300 klip; 3.059 klip berlabel dari 244 peserta (34 positif) setelah dedup.
- `kalibrasi-data-nyata.md`: pada audio nyata W=22 sudah lolos ambang K-04 (0,029 oktaf), W=24 0,007; W=18 gagal (0,79).
  Usulan `FEAT_OFFSET` 8265 (jenuh 0,2%) menggantikan 6690. params.py BELUM diubah, menunggu keputusan pengguna.
- Baseline awal (perekam saja, 244 peserta) DIHAPUS atas permintaan pengguna; hanya model lengkap (perekam + Phone A, `model.md`) yang dipakai. Temuan awal: AUROC peserta 0,41-0,43, model sederhana ≈ label-acak.
- Berkas baru belum di-commit: `golden/calibrate_real.py`, `model/{build_data,baseline,mil_train,fusion,describe_data}.py`, dokumen vault.

## Ringkasan 2026-10-11 (setelah Fase M/1/2)
- Data: perekam + Phone A diunduh (397/398 berkas; 1 tanpa ekstensi `.wav` dilewati); 664 rekaman -> 7.187 klip berlabel,
  360 peserta, 58 positif. Phone A berformat 192 kHz stereo (di-resample ke 16 kHz).
- `params.FEAT_OFFSET` diubah 6690 -> 8265 (K-14, data nyata); tabel/paket SV/vektor diregenerasi.
- Model: AUROC batuk saja 0,562 (CI 0,48-0,64), nol permutasi rata-rata 0,523 (p95 0,591); fusi gejala 0,746 vs gejala saja 0,743.
  Logit model akhir hampir konstan (-1..8 pada Q8): TIDAK ADA bukti sinyal; jangan membuat klaim akurasi klinis.
- Fase 1: Makefile, `rtl/common` (rom/ram/fifo/sva), `rtl/files.f`, `tb/common` (vec, cocotb_utils, runner), `docs/vault/verifikasi.md`.
- Fase 2: `rtl/audio/{i2s_rx,preemph,frame_buf}.sv`, `rtl/detect/cough_det.sv`, 4 testbench cocotb, vektor golden. BELUM dilint
  maupun disimulasikan (aturan Fase 1-5). Disarankan `make lint TOP=<modul>` sebelum Fase T untuk menangkap kesalahan sintaks.
- Semua perubahan belum di-commit (sesuai permintaan).

## Pembaruan remodel (2026-10-11 pagi)
- Remodel butir 1-3 dikerjakan (ringkasan di `model.md`): CNN diperbaiki 0,45-0,49; LR statistik peserta 0,66-0,70;
  perancu jenis kelamin menjelaskan sebagian besar (sex saja 0,65; audio -> sex 0,89; audio residu 0,58).
- Unduhan: Phone B/C berjalan (tmux `kaggle2`); TBscreen WAV + metadata (1,37 GB dari zip UW 424 GB, tmux `tbscreen`).
  CODA TB DREAM dibatalkan (K-15).
- Keputusan arsitektur Fase 4 tertunda: CNN vs akumulator statistik + LR (lihat `model.md` dan K-17).

## Pembaruan langkah 1-3 (2026-10-11 siang)
- TBscreen terunduh (WAV + CSV, 1,37 GB; 28.366 berkas, CRC lolos) dan diproses: `model/tbscreen.py`, `model/tbs_eval.py`,
  hasil di `model.md` bagian 5. Tidak ada sinyal TB yang andal; cohort A/C hampir = label (perancu rekrutmen).
- Fitur klinis diperluas + fusi bersarang: `model/clinical.py`, `model.md` bagian 4. Fusi terbaik 0,768 (target 0,80 belum).
- Uji embedding VGGish: `model/embed_test.py`, `model/embed_eval.py`; hasil di `model.md` bagian 5. Distilasi dibatalkan (K-16).
- Belum ada commit. Keputusan tertunda: revisi target (K-16), arsitektur Fase 4 (CNN vs akumulator + LR).
- Dokumen model dilebur ke satu berkas `model.md` (7 laporan lama dipindah ke `$SUARATB_DATA/runs/reports/`, di luar repo).
  Arsitektur Fase 4 diputuskan: CNN output-stationary (K-17). Phone B/C Kaggle masih diunduh (tmux `kaggle2`).

## Fase 3 ditulis (2026-10-11)
- RTL: `rtl/dsp/{window,fft512,power,mel,log_q,quant,dsp_chain,feat_buf}.sv`; `rtl/files.f` diperbarui.
- TB: `tb/test_{window,fft512,power,mel,log_q,quant,dsp_chain,feat_buf}.py`, `tb/common/drive.py`; vektor memakai
  `tb/vectors/{window,fft512,power,mel,log_q,quant,frontend}` (sudah identik dengan keluaran `gen_vectors`).
- Model Python jadwal/alamat: `golden/test_fft_schedule.py`, `golden/test_dsp_addressing.py` (67 tes golden lulus).
- Tes unit Fase 2 dan 3 dijalankan (Icarus, K-19): 12 modul LULUS bit-exact terhadap golden. Berikutnya: Fase 4 (CNN, K-17).

## Fase 4 ditulis dan dites unit (2026-10-11)
- RTL `rtl/cnn/*.sv`, TB `tb/test_{requant,gap,cnn_top,cnn_sweep,cnn_trained}.py`, generator `golden/gen_cnn_rom.py`.
- cnn_top bit-exact vs golden (bobot acak, bobot terlatih model_v2, P=8/16/32). Siklus: 451.851 / 226.490 / 122.843.
- Berikutnya: Fase 5 (top, csr, uart, mode dataset/mikrofon), lalu Fase T dan 6.
