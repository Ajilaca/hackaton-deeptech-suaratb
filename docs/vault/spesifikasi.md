# Spesifikasi final (Fase 0)

Sumber tunggal: `golden/params.py`. Berkas ini hanya ringkasan; jika berbeda, `params.py` yang benar.
Status data: **`CALIBRATION_SOURCE = "synthetic"`**. Konstanta bertanda ⚠ bergantung pada data dan
PROVISIONAL sampai `python -m golden.calibrate --data data/raw` dijalankan pada dataset nyata.
Alasan tiap nilai: `keputusan.md` (K-xx).

## Audio dan framing
| Parameter | Nilai | Catatan |
|---|---|---|
| Laju sampel | 16 kHz, 16 bit | I2S INMP441; clock inti 98,304 MHz, BCLK = clk/96 |
| Frame / hop / FFT | 400 / 160 / 512 | 25 ms / 10 ms; zero-pad 112 |
| Segmen fitur | 50 frame x 40 mel | 5 pre-roll + 45 pasca-onset (K-03) |
| Klip dataset | 8000 sampel + 240 nol = 8240 | tepat 50 frame |

## Front-end (bit-exact di `golden/blocks/`)
| Blok | Definisi | Parameter beku |
|---|---|---|
| preemph | y = x - 0,97 x[n-1]; keluaran 16+8 = 24 bit (tanpa pembulatan int16, K-08) | PREEMPH_Q15 = 31785, PREEMPH_FRAC = 8 |
| window | Hamming 400 titik Q1.15, keluaran W bit | WIN_FRAC = 15 |
| fft512 | radix-2 DIT, real, skala 1/2 per stage (total 1/512), twiddle Q1.15, satu pembulatan per butterfly | **FFT_W = 24** (K-04), FFT_STAGE_SHIFT = (1,)x9 |
| power | re² + im², bin 0..256, 47 bit tak bertanda | POWER_BITS = 47 |
| mel | 40 band HTK 50-8000 Hz, bobot Q0.12 sparse (493 entri ROM, maks. 2 band/bin) | MEL_W_FRAC = 12, MEL_SHIFT = 0, akumulator 58 bit |
| log_q | leading-one + LUT 256 entri; keluaran Q6.8 (14 bit) (K-05) | LOG_IN_BITS = 58, LOG_FRAC = 8 |
| quant | q = sat8(round((L - OFFSET) / 2^SHIFT)) | FEAT_SHIFT = 5, ⚠ FEAT_OFFSET = 8265 (K-14, data nyata perekam) |

## CNN int8
| Parameter | Nilai |
|---|---|
| Arsitektur | Conv3x3 same (16/32/32) -> requant -> ReLU -> MaxPool2x2, GAP, FC 32->16 (ReLU), FC 16->1 |
| Bobot / bias / MAC | 14.496 / 97 / 3.698.448 (diverifikasi `golden/test_params.py`) |
| Akumulator | 24 bit bertanda; batas terburuk 288x127x128 = 4.681.728 < 2^23 (+margin bias ~3,7 juta) (K-10) |
| Requant | y = (acc·M0 + 2^(s-1)) >> s; M0 Q15 tak bertanda; ReLU+saturasi [0,127]; FC akhir int16 tanpa ReLU |
| GAP | sum · 1092 (=round(2^15/30)) + 2^14 >> 15 atas peta 5x6 |
| Kanal / P | C1,C2,C3 = 16,32,32; FC_HIDDEN = 16; P = 16 (8/16/32); semuanya parameter |
| Nilai requant & bobot | **belum ada**: berasal dari Fase M (QAT). Golden hanya punya `random_model` + `calibrate_requant` untuk uji |

## Detektor batuk ⚠ (sintetis)
E[f] = Σ x_pe² atas 160 sampel terbaru frame f; N: EMA 2^-6 dengan 6 bit pecahan; trigger E > 8·max(N, 640)
selama ≥ 2 frame; segmen = onset-5 .. onset+44; N dibekukan saat merekam; refractory 30 frame (300 ms) (K-07).

## Estimasi hardware (BUKAN fakta; sumber `docs/vault/anggaran-fase0.md`)
| Besaran | Nilai | Dasar |
|---|---|---|
| Memori logis (P=16) | 288.936 bit (282 Kbit) | hitungan analitik |
| Siklus CNN ideal P=8/16/32 | 462.320 / 231.168 / 124.608 | analitik, tanpa overhead |
| Blok pengali XC7A35T (P=16) | 31 dari 90 | analitik, tanpa sintesis |
| BRAM36 XC7A35T | ~17 dari 50 | per-memori, estimasi |
| LUT | 4-8 rb (dari CLAUDE.md) | tebakan, keyakinan rendah |
| Fmax | target 100 MHz, tantangan 150 MHz | belum diukur |

## Antarmuka sistem (Fase 5, `rtl/sys/`)
| Item | Nilai |
|---|---|
| Masukan sampel | mode 1: I2S INMP441 (FPGA master, BCLK clk/96); mode 0: UART dataset |
| UART | 8N1, `CLK_PER_BIT` = clk/baud (853 -> 115200 pada 98,304 MHz); tanpa FIFO: 2 byte/sampel harus >= 404 siklus |
| Perintah host -> alat | `'W' addr data` tulis CSR; `'R' addr` balasan 1 byte; `'D'` + 8240 x int16 LE = klip dataset |
| Hasil alat -> host | `'L' lo hi seq` (4 byte) tiap inferensi: logit int16 per batuk (rata-rata 3-5 batuk di host) |
| CSR | 0x00 CTRL (bit0 mode, bit1 clear), 0x01 STATUS, 0x02-03 LOGIT, 0x04 SEQ, 0x05-06 NSAT, 0x07 VERSION, 0x08-09 FRAMES |
| Alur | sampel -> preemph -> frame_buf -> dsp_chain -> feat_buf -> cnn_top -> logit; frame_buf.e -> cough_det -> segmen |
