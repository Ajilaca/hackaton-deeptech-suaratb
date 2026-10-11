# Sintesis (Fase 6)

Satu-satunya dokumen hasil sintesis. Angka berasal dari laporan alat (Vivado 2026.1 dan Yosys 0.33, 2026-10-11); yang
dihitung dari angka lain diberi label.

## Metode
- Referensi: Vivado 2026.1, **XC7A35T-1** (`xc7a35tcpg236-1`), `synth_design -mode out_of_context`, lalu `opt_design`,
  `place_design`, `route_design`. Clock 10 ns (100 MHz) pada port `clk`; **tanpa constraint I/O dan tanpa MMCM** (tidak ada pin).
  Timing = jalur register ke register dalam inti.
- Model: bobot terlatih `model_v2` (ROM isi nyata), P = 8 / 16 / 32 (paket konfigurasi dibangkitkan dari `golden/gen_cnn_rom.py`).
- Yosys: `synth_xilinx -family xc7` setelah `sv2v` (Yosys 0.33 tidak membaca `import pkg::*` dan `return` dalam fungsi); tanpa
  place and route, tanpa timing.
- Skrip: `target/artix7/{prep_synth.py,synth.tcl,run_sweep.sh}`, `target/yosys/synth.sh`. Laporan lengkap di
  `$SUARATB_DATA/build/vivado/P*/` dan `build/yosys/P*/`.

## Hasil Vivado (setelah route, suaratb_top lengkap, constraint clock 10 ns)

XC7A35T: 20.800 LUT, 41.600 FF, 90 DSP48E1, 50 BRAM36.

| P | LUT | FF | DSP48E1 | BRAM tile (RAMB36 / RAMB18) | WNS @ 10 ns | Daya total* |
|---|---|---|---|---|---|---|
| 8 | 4.823 (23,2%) | 5.548 (13,3%) | 17 (18,9%) | 12 (9 / 6) = 24% | +0,443 ns | 0,138 W |
| 16 | 5.834 (28,1%) | 6.148 (14,8%) | 17 (18,9%) | 12 (9 / 6) = 24% | +0,077 ns | 0,149 W |
| 32 | 8.236 (39,6%) | 7.486 (18,0%) | 17 (18,9%) | 12 (9 / 6) = 24% | +0,258 ns | 0,182 W |

\* Estimasi tanpa aktivitas switching dari simulasi (vectorless), keyakinan "Medium" menurut alat; bukan pengukuran.

**Timing 100 MHz terpenuhi untuk P = 8, 16, 32 (WNS > 0, 0 endpoint gagal).**

Fmax (satu jalankan per titik; perkiraan konservatif 1 / (T - WNS), T = constraint):

| P | Constraint | WNS | Periode tercapai | Fmax perkiraan | Jalur kritis |
|---|---|---|---|---|---|
| 8 | 8 ns | -1,378 ns | 9,38 ns | **≈ 106,6 MHz** | `u_ramB (BRAM) -> u_pe/prod_reg` (baca BRAM + pengali) |
| 16 | 8 ns | -1,240 ns | 9,24 ns | **≈ 108,2 MHz** | `u_ramA (BRAM) -> u_pe/prod_reg` |
| 16 | 6,67 ns | -2,658 ns | 9,33 ns | ≈ 107,2 MHz | `u_ramB (BRAM) -> u_pe/prod_reg` |
| 32 | 8 ns | -1,547 ns | 9,55 ns | **≈ 104,7 MHz** | `u_ctrl/iy1 -> u_ramB/ADDR` (alamat buffer B) |

Fmax P = 8 / 16 / 32 ≈ 105-108 MHz: **100 MHz tercapai untuk semua P; 150 MHz tidak tercapai**. Perbedaan antar-P kecil dan
berada dalam variasi penempatan (satu jalankan, tanpa seed sweep).

### Perbaikan yang dibuat dari hasil sapuan pertama (RTL, tetap bit-exact)
Sapuan pertama (sebelum perbaikan): P=16 WNS -0,440 ns (Fmax ≈ 96 MHz), jalur kritis di kupu-kupu FFT (`fft512`: penjumlahan lebar +
penggeser variabel + saturasi + penghitung dalam satu siklus). Perbaikan:
1. `rtl/dsp/fft512.sv`: `a*2^15 + rnd` dihitung satu tahap lebih awal; geser 15/16 menjadi pemilih 2:1; penghitung saturasi
   dipecah (sat_inc di t+4, tambah ke sat_cnt di t+5). Latensi butterfly tetap; n_sat identik.
2. `rtl/detect/cough_det.sv`: dipecah 3 tahap (tangkap e; N, K*N, EMA; bandingkan + perbarui) karena jalur kritis berikutnya
   di `u_det/nq_reg -> run_reg` (carry 64 bit berantai). Latensi 1 -> 3 siklus (aman: 1 e_valid per frame = ribuan siklus).
Setelah perbaikan: `test_fft512`, `test_dsp_chain`, `test_cough_det`, `test_feat_buf`, `test_frame_buf` lulus; `test_top`
mode mikrofon (`test_mic`) bit-exact dengan golden. Regresi penuh `test_top` (dataset + mikrofon) dijalankan di Fase T.

- **DSP48E1 = 17 untuk semua P**: pengali 8 x 8 di PE array dipetakan ke LUT/CARRY oleh Vivado, bukan ke DSP; DSP dipakai
  FFT/power/mel. Karena itu kenaikan P menambah LUT, bukan DSP. Tidak ada atribut vendor di `rtl/` (aturan 5).
- BRAM tidak bergantung pada P (buffer A/B, ROM bobot, RAM FFT, feat_buf).

## Latensi CNN (siklus dari simulasi x periode; angka turunan, bukan laporan alat)

| P | Siklus inferensi (simulasi) | @ 98,304 MHz (rencana I2S) | @ 100 MHz (timing terpenuhi) |
|---|---|---|---|
| 8 | 451.851 | 4,60 ms | 4,52 ms |
| 16 | 226.490 | 2,30 ms | 2,26 ms |
| 32 | 122.843 | 1,25 ms | 1,23 ms |

Latensi yang dirasakan pengguna tetap didominasi pengambilan 0,5 s audio.

## Hasil Yosys (perbandingan lintas-vendor; tanpa timing)

| P | LUT | FF | DSP48E1 | BRAM36 / BRAM18 |
|---|---|---|---|---|
| 8 | 11.083 | 6.358 | 32 | 9 / 2 |
| 16 | 11.507 | 6.684 | 40 | 9 / 2 |
| 32 | 12.560 | 7.403 | 56 | 9 / 2 |

Yosys memetakan lebih banyak LUT dan DSP daripada Vivado (LUT 1,5-2,4x; DSP 1,9-3,3x) dan lebih sedikit BRAM; angka Yosys bukan
prediksi Vivado dan tidak memuat timing. Selisih ini wajar untuk dua alat yang berbeda optimasi dan packing.

## Dibandingkan dengan estimasi awal (CLAUDE.md)

| Besaran | Estimasi | Vivado P=16 |
|---|---|---|
| LUT | 4-8 rb | 5,8 rb (di dalam rentang) |
| DSP48 | 25-30 | 17 (di bawah estimasi) |
| BRAM36 | 15-25 | 12 tile (9 RAMB36 + 6 RAMB18; di bawah estimasi) |
| Fmax | target 100 MHz, tantangan 150 MHz | ≈ 105-108 MHz untuk P = 8/16/32 (100 MHz tercapai; 150 MHz belum) |

## Batasan
- OOC tanpa constraint I/O: Fmax jalur I/O dan clock board belum diuji. Hanya satu jalankan per P (tanpa seed sweep).
- Bukan papan; tidak ada klaim "jalan di papan". Papan panitia belum diumumkan (target/ belum ada pin).
- Daya vectorless, bukan pengukuran.

## Langkah berikut
1. Bila 150 MHz dibutuhkan: tambah register keluaran BRAM dan pecah pengali 8 x 8 di `pe_array` (jalur kritis sekarang).
2. Beberapa seed untuk rentang Fmax; constraint I/O dan clock board setelah papan diumumkan.
3. Angka sumber daya per modul (`util_impl_hier.rpt`) untuk tabel proposal.
