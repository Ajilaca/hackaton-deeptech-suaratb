# Kalibrasi Fase 0 pada audio NYATA (perekam, Zambia CIDRZ)

Dihasilkan `python -m golden.calibrate_real` dari 1764 klip batuk (hasil detektor golden) pada 150 rekaman unik. **params.py tidak diubah**; angka di bawah adalah masukan keputusan.

## Tingkat sinyal
RMS per hop (LSB int16): p10/p50/p90/p99 = 6/15/78/1103
Puncak klip: median 3955, maks 32768 (skala penuh 32767). Batas atas akumulator mel teramati: 49 bit (batas analitik 58).

## Lebar FFT: galat fitur vs float (oktaf, elemen ≥ lantai derau kuantisasi + 10 dB) dan SNR FFT
| W | galat rata-rata | galat p90 klip | fraksi elemen dinilai | SNR FFT rata-rata (dB) | SNR p10 (dB) | saturasi |
|---|---|---|---|---|---|---|
| 16 | 2.290 | 3.605 | 1.00 | 18.3 | 1.6 | 0 |
| 18 | 0.788 | 1.494 | 1.00 | 29.9 | 12.7 | 0 |
| 20 | 0.123 | 0.236 | 1.00 | 42.0 | 24.8 | 0 |
| 22 | 0.029 | 0.052 | 1.00 | 54.1 | 36.9 | 0 |
| 24 | 0.007 | 0.013 | 1.00 | 65.6 | 49.0 | 0 |

Ambang keputusan K-04: galat ≤ 0,0625 oktaf (0,5 langkah int8).

## Jadwal skala (W=24) pada 60 klip berpuncak tertinggi
| jadwal | sampel jenuh | klip jenuh |
|---|---|---|
| semua 1/2 | 0 | 0 |
| 1 stage tanpa skala | 0 | 0 |
| 2 stage | 1 | 1 |
| 3 stage | 27 | 11 |

## Kuantisasi fitur (W=24, L Q6.8)
persentil L (0,5/1/5/50/95/99/99,5): 5320 / 5503 / 6033 / 7609 / 10242 / 10973 / 11210
- nilai berjalan (sintetis): OFFSET 6690, SHIFT 5: jenuh int8 2.02%
- usulan dari data nyata: **OFFSET 8265, SHIFT 5**: jenuh int8 0.20%; elemen nol 0.200%

## Detektor: laju kejadian pada rekaman utuh (tanpa ground truth)
| K | EMA S | median kejadian/rekaman | kejadian/menit | rekaman tanpa kejadian |
|---|---|---|---|---|
| 4 | 5 | 18 | 24.4 | 0% |
| 4 | 6 | 17 | 23.1 | 0% |
| 4 | 7 | 16 | 21.3 | 0% |
| 6 | 5 | 13 | 18.3 | 0% |
| 6 | 6 | 13 | 18.1 | 0% |
| 6 | 7 | 12 | 17.5 | 0% |
| 8 | 5 | 11 | 15.1 | 0% |
| 8 | 6 | 11 | 15.3 | 0% |
| 8 | 7 | 11 | 15.3 | 0% |
| 12 | 5 | 8 | 11.8 | 0% |
| 12 | 6 | 9 | 12.4 | 0% |
| 12 | 7 | 9 | 12.5 | 0% |
| 16 | 5 | 7 | 10.0 | 1% |
| 16 | 6 | 8 | 10.6 | 0% |
| 16 | 7 | 8 | 11.0 | 0% |
