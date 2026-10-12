# Status fase

**Status (2026-10-12): Fase 0-6 selesai; versi 150 MHz (`optimized`) dan remodeling model (`remodel`) selesai. Fase T selesai
(regresi terpadu 20/20 lulus di Icarus). Sisa: Fase 7 (papan, bila diumumkan), Fase 8 (benchmark CPU dan angka akhir proposal).**

| Fase | Status | Hasil utama |
|---|---|---|
| 0 | Selesai | Parameter dibekukan di `golden/params.py`; 67 tes golden lulus; kalibrasi pada audio nyata (K-14) |
| M | Selesai (pipeline); hasil klinis audio dari-nol negatif | Kuantisasi int8 turun 0,001; model dari-nol ~0,56; lihat `model.md` |
| 1-2 | Selesai | Infrastruktur verifikasi; `i2s_rx`, `preemph`, `frame_buf`, `cough_det` bit-exact |
| 3 | Selesai | `window`, `fft512`, `power`, `mel`, `log_q`, `quant`, `feat_buf` bit-exact |
| 4 | Selesai | CNN output-stationary, P = 8/16/32, bit-exact (bobot acak dan terlatih) |
| 5 | Selesai | `suaratb_top`: mode dataset (UART) dan mikrofon (I2S), bit-exact ujung ke ujung |
| 6 | Selesai | Vivado XC7A35T-1: 100 MHz terpenuhi (Fase 6) dan 150 MHz (`optimized`, K-22); Yosys sebagai pembanding; `sintesis.md` |
| Remodel | Selesai (eksploratori) | Embedding HeAR/OPERA, protokol evaluasi tetap, uji permutasi, distilasi ke CNN golden; K-23, `model.md` §5d-5e |
| T | Selesai | `make test FASE_T=1 SIM=icarus` (2026-10-12): 20 tes `tb/` lulus, 0 gagal, 45 menit 45 detik; diulang setelah penataan direktori `golden/` dan `tb/`: 20 lulus, 41 menit 5 detik, exit code 0; 67 tes `golden/` lulus. Model: bobot acak terkalibrasi (`test_top` bobot distilasi diuji terpisah). Hanya simulasi Icarus, bukan papan; Verilator tidak dipakai (cocotb 2.1 tidak cocok dengan Verilator 5.020) |
| 7 | Belum | Papan panitia belum diumumkan |
| 8 | Belum | Benchmark vs CPU dan angka akhir |

Cabang git (berantai): `fase0-2` -> `fase3` -> `fase4` -> `fase5` -> `fase6` -> `optimized` -> `remodel` -> `finalisasi`; `main` mengikuti cabang terbaru yang digabung.

## Catatan kronologis (riwayat pembaruan)

## Sebelum Fase 1 / yang masih terbuka
1. `data/raw` kosong: ulangi `python -m golden.analysis.calibrate --data data/raw` untuk FEAT_OFFSET, detektor, dan AUROC
   (syarat "AUROC turun ≤ 0,01" untuk lebar FFT belum diperiksa).
2. Requant/bobot CNN: dari Fase M (QAT); golden hanya punya model acak + kalibrator.
3. Tafsiran segmen 50 frame (K-03) dan jadwal skala FFT `first1_0` (K-04): perlu konfirmasi.
4. Kapasitas perangkat dan jenis FPGA panitia: dari ingatan / belum diumumkan.
5. Mikrofon INMP441 mengeluarkan 24 bit; spesifikasi memakai 16 bit teratas. Bila bit bawah dipakai, derau
   kuantisasi masukan turun dan lebar FFT mungkin perlu dihitung ulang (tidak dievaluasi).
6. Makefile belum ada (Fase 1). Perintah: `.venv/bin/python -m pytest golden`, `-m golden.gen.gen_tables`,
   `-m golden.gen.gen_vectors`, `-m golden.analysis.budget`, `-m golden.analysis.calibrate`.

## Pembaruan data nyata (2026-10-11)
- `data/raw` -> `/data/suaratb/raw`; perekam (281 WAV) selesai, Phone A (3 klinik, 26,3 GB) masih diunduh.
- Segmentasi perekam: 280 rekaman -> 3.300 klip; 3.059 klip berlabel dari 244 peserta (34 positif) setelah dedup.
- `kalibrasi-data-nyata.md`: pada audio nyata W=22 sudah lolos ambang K-04 (0,029 oktaf), W=24 0,007; W=18 gagal (0,79).
  Usulan `FEAT_OFFSET` 8265 (jenuh 0,2%) menggantikan 6690. params.py BELUM diubah, menunggu keputusan pengguna.
- Baseline awal (perekam saja, 244 peserta) DIHAPUS atas permintaan pengguna; hanya model lengkap (perekam + Phone A, `model.md`) yang dipakai. Temuan awal: AUROC peserta 0,41-0,43, model sederhana ≈ label-acak.
- Berkas baru belum di-commit: `golden/analysis/calibrate_real.py`, `model/{build_data,baseline,mil_train,fusion,describe_data}.py`, dokumen vault.

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
- TBscreen terunduh (WAV + CSV, 1,37 GB; 28.366 berkas, CRC lolos) dan diproses: `model/data/tbscreen.py`, `model/evaluation/tbs_eval.py`,
  hasil di `model.md` bagian 5. Tidak ada sinyal TB yang andal; cohort A/C hampir = label (perancu rekrutmen).
- Fitur klinis diperluas + fusi bersarang: `model/clinical_models/clinical.py`, `model.md` bagian 4. Fusi terbaik 0,768 (target 0,80 belum).
- Uji embedding VGGish: `model/pretrained/embed_test.py`, `model/pretrained/embed_eval.py`; hasil di `model.md` bagian 5. Distilasi dibatalkan (K-16).
- Target 0,70/0,80 diganti pelaporan terukur + pembanding WHO TPP (K-21). Klinis saja 0,772 (1.828 peserta); audio tidak
  menambah (+0,003). Lintas-perangkat Phone B/C 0,52-0,63. Detail di `model.md` bagian 5a-5c.
- Fase 5: `test_top` (dataset 2 klip + mikrofon) LULUS setelah dua perbaikan RTL: pembersihan jalur DSP saat ganti mode, dan
  menunggu frame ke-50 dari awal segmen (bukan frame ke-45) di `suaratb_top.sv`.
- Cabang lokal: fase0-2 (5 commit), fase3, fase4, fase5-6 (belum ada commit). Push oleh pengguna.
- Dokumen model dilebur ke satu berkas `model.md` (7 laporan lama dipindah ke `$SUARATB_DATA/runs/reports/`, di luar repo).
  Arsitektur Fase 4 diputuskan: CNN output-stationary (K-17). Phone B/C Kaggle masih diunduh (tmux `kaggle2`).

## Fase 3 ditulis (2026-10-11)
- RTL: `rtl/dsp/{window,fft512,power,mel,log_q,quant,dsp_chain,feat_buf}.sv`; `rtl/files.f` diperbarui.
- TB: `tb/test_{window,fft512,power,mel,log_q,quant,dsp_chain,feat_buf}.py`, `tb/common/drive.py`; vektor memakai
  `tb/vectors/{window,fft512,power,mel,log_q,quant,frontend}` (sudah identik dengan keluaran `gen_vectors`).
- Model Python jadwal/alamat: `golden/tests/test_fft_schedule.py`, `golden/tests/test_dsp_addressing.py` (67 tes golden lulus).
- Tes unit Fase 2 dan 3 dijalankan (Icarus, K-19): 12 modul LULUS bit-exact terhadap golden. Berikutnya: Fase 4 (CNN, K-17).

## Fase 4 ditulis dan dites unit (2026-10-11)
- RTL `rtl/cnn/*.sv`, TB `tb/test_{requant,gap,cnn_top,cnn_sweep,cnn_trained}.py`, generator `golden/gen/gen_cnn_rom.py`.
- cnn_top bit-exact vs golden (bobot acak, bobot terlatih model_v2, P=8/16/32). Siklus: 451.851 / 226.490 / 122.843.
- Berikutnya: Fase 5 (top, csr, uart, mode dataset/mikrofon), lalu Fase T dan 6.

## Fase 5-6, optimized, dan remodel (2026-10-11 s.d. 12)
- Fase 6: Vivado P = 8/16/32 pada 100 MHz terpenuhi setelah perbaikan timing FFT dan `cough_det`; Yosys 0,33 via sv2v sebagai pembanding.
- `optimized`: pipeline ulang CNN, DSP, front-end, detektor; 150 MHz terpenuhi (WNS +0,111 / +0,222 / +0,023 ns untuk P = 8/16/32);
  bit-exact (`test_top`); K-22 (clock inti 147,456 MHz).
- Remodel: audit data (364 peserta/60 positif, rilis Kaggle lengkap), protokol evaluasi tetap, HeAR/OPERA, uji permutasi statistik
  maksimum, lintas-perangkat, leave-one-clinic-out, distilasi HeAR ke CNN golden (0,654 vs guru 0,672), kuantisasi int8 lulus, RTL
  bit-exact dengan bobot murid. Keputusan: K-23. Kode dirapikan ke subpaket `model/{data,scratch,clinical_models,evaluation,pretrained,distillation}`.
