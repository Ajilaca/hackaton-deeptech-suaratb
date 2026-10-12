# SuaraTB

Akselerator skrining TBC dari suara batuk untuk Hackathon Deeptech 2026 (Indonesia Chip Hackathon Challenge, ITB/ICDeC), tema
Medical Device. Alur: mikrofon I2S -> fitur log-mel -> CNN int8 -> skor risiko. Audio tidak keluar dari perangkat. RTL
vendor-neutral; model numpy bit-exact (`golden/`) adalah spesifikasi hardware.

> **Batas klaim.** Alat ini adalah pendukung triase/prioritas rujukan ke tes TCM/dahak, **bukan** diagnosis. Hasil terukur
> belum memenuhi standar WHO TPP triase (sensitivitas >= 90%, spesifisitas >= 70%). Akurasi audio belum terbukti secara presisi
> (lihat bagian Hasil). Tidak ada klaim "jalan di papan": semua angka hardware berasal dari simulasi dan laporan sintesis.

## Struktur repo
| Jalur | Isi |
|---|---|
| `golden/` | Spesifikasi: model numpy bit-exact, parameter beku (`params.py`), pembangkit tabel/vektor/anggaran, tes golden |
| `rtl/` | SystemVerilog vendor-neutral: audio, detektor batuk, DSP, CNN, sistem (UART, CSR, top) |
| `tb/` | Testbench cocotb per modul dan ujung ke ujung, vektor uji sintetis dari golden |
| `target/` | Satu-satunya kode spesifik perangkat: skrip sintesis Vivado (Artix-7) dan Yosys |
| `model/` | Data, pelatihan, evaluasi, embedding pralatih, distilasi, ekspor bobot int8 |
| `sw/` | Penampung perangkat lunak host (fusi klinis, benchmark, aplikasi demo); belum berisi kode |
| `docs/` | Konteks proyek (`vault/`), keputusan, dan laporan hasil (`result/`) |

Data, audio, fitur, bobot, dan proyek Vivado disimpan di luar repo (`$SUARATB_DATA`, awal `/data/suaratb`) dan tidak di-commit.

## Spesifikasi ringkas
| Bagian | Nilai |
|---|---|
| Audio | 16 kHz, 16 bit, mikrofon I2S (INMP441), klip 0,5 s |
| Front-end | Pre-emphasis 0,97, frame 400/hop 160, Hamming, FFT 512 radix-2 (data 24 bit), mel 40 band, log2 LUT, int8 |
| Detektor batuk | Energi per hop, EMA derau, ambang 8x, tahan 300 ms |
| CNN | Conv3x3 16/32/32 + ReLU + MaxPool, GAP, FC 32->16->1; 14.496 bobot int8, 3.698.448 MAC |
| Akselerator | Output-stationary, P lane paralel (8/16/32), akumulator 24 bit, requant per layer |
| Antarmuka | I2S masuk, UART keluar (skor) dan mode dataset lewat UART, CSR 8 bit |

## Status
Fase 0-6 selesai, versi 150 MHz (`optimized`) dan remodeling model selesai. Sisa: regresi terpadu (Fase T), papan (Fase 7, belum
diumumkan), benchmark CPU (Fase 8). Rincian: `docs/vault/status-fase.md`.

## Hasil (apa adanya)
| Aspek | Hasil | Sumber |
|---|---|---|
| Kesesuaian RTL dengan golden | Bit-exact per modul dan ujung ke ujung (dataset dan mikrofon), P = 8/16/32 | `docs/vault/verifikasi.md` |
| Timing Vivado XC7A35T-1 (OOC) | 100 MHz terpenuhi; 150 MHz terpenuhi untuk P = 8/16/32 (WNS +0,023 s.d. +0,222 ns) | `docs/vault/sintesis.md` |
| Sumber daya, P = 16 @ 150 MHz | 5.926 LUT (28,5%), 7.799 FF, 17 DSP48E1, 12 tile BRAM, 0,19 W (vectorless) | `docs/vault/sintesis.md` |
| Kuantisasi int8 | Penurunan AUROC 0,001 (model dari nol), 0,008 (murid distilasi); kriteria <= 0,02 | `docs/vault/model.md` |
| Gejala klinis saja | AUROC 0,77 (1.828 peserta, 205 positif) | `docs/vault/model.md` §5b |
| Audio, CNN dilatih dari nol | AUROC ~0,56; tak dapat dibedakan dari nol permutasi | `docs/vault/model.md` §4 |
| Audio, embedding HeAR (364 peserta, 60 positif) | AUROC 0,71 dalam jenis kelamin, p disesuaikan 0,002; tambahan di atas klinis +0,04 dengan CI mencakup 0 | `docs/vault/model.md` §5d |
| Distilasi HeAR -> CNN golden | AUROC 0,654 (guru 0,672); RTL bit-exact dengan bobot ini | `docs/vault/model.md` §5e |
| Standar WHO TPP | Tidak dipenuhi | `docs/vault/model.md` §5c |

Laporan lengkap: `docs/result/result.md` (dan `result.pdf`).

## Dokumen
Baca berurutan: `docs/vault/status-fase.md`, `docs/vault/spesifikasi.md`, `docs/vault/keputusan.md`. Indeks: `docs/CONTEXT_INDEX.md`.
Perintah build dan tes ada di `Makefile` (`make help`) dan docstring skrip; README tidak mengulangnya.

## Lisensi dan data
Lihat `LICENSE`. Data Zambia CIDRZ (ODbL-1.0) dan TBscreen (CC BY 4.0) tidak didistribusikan di repo ini. Repo privat sampai
setelah Demo Day.
