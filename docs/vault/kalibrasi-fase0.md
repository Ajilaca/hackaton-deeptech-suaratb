# Hasil kalibrasi Fase 0 (audio SINTETIS)

Dihasilkan oleh `python -m golden.analysis.calibrate` (mentah: `golden/out/phase0_calibration.json`, tidak di-commit).
Generator: `golden/gen/synth.py` (derau berwarna + letupan berformant bukan batuk klinis). Angka ini menunjukkan
perilaku aritmetika, BUKAN kinerja klinis, dan harus diulang pada dataset nyata.

## Ensemble
5 tingkat derau latar (σ = 1,5 / 3 / 10 / 30 / 100 LSB int16) x 3 seed x {batuk 10 dB, batuk 25 dB, derau, bicara} = 60 klip 0,5 s.
Lantai derau kuantisasi masukan (putih, var 1/12 LSB²) dihitung per band (`frontend.input_floor_L`); galat fitur hanya
dihitung di elemen ≥ lantai + 10 dB karena di bawahnya fitur hanyalah derau kuantisasi int16.

## FFT: SNR (dB, FFT golden vs FFT float dari masukan window yang sama) dan galat fitur (oktaf)
| W | σ=1,5 | σ=3 | σ=10 | σ=30 | σ=100 |
|---|---|---|---|---|---|
| 16 | -14,6 / 5,97 | -11,1 / 5,79 | -3,3 / 4,25 | 5,1 / 2,62 | 15,8 / 1,28 |
| 18 | -6,4 / 3,42 | -1,5 / 2,94 | 7,2 / 2,20 | 17,1 / 1,08 | 28,0 / 0,27 |
| 20 | 3,8 / 0,72 | 9,4 / 0,61 | 19,3 / 0,48 | 29,3 / 0,15 | 40,1 / 0,053 |
| 22 | 15,8 / 0,125 | 21,7 / 0,116 | 31,5 / 0,085 | 41,3 / 0,039 | 52,1 / 0,013 |
| 23 | 21,8 / 0,065 | 27,7 / 0,058 | 37,5 / 0,044 | 47,3 / 0,020 | 58,1 / 0,006 |
| 24 | 27,9 / 0,033 | 33,8 / 0,030 | 43,6 / 0,023 | 53,4 / 0,010 | 64,0 / 0,004 |
| 25 | 34,0 / 0,017 | 39,8 / 0,015 | 49,6 / 0,011 | 59,4 / 0,005 | 69,8 / 0,002 |

Tidak ada saturasi pada jadwal 1/2 per stage di seluruh ensemble. Derau pembulatan FFT ≈ 0,58 LSB absolut
(satuan guard), jadi SNR naik ±6 dB per bit data pada sinyal kecil dan mendatar di ~85 dB pada sinyal keras
(batas twiddle Q1.15).

## Jadwal skala (W=24, SNR FFT dB; jumlah sampel jenuh)
| Jadwal | derau 3 LSB | batuk rms 1000 | rms 3000 | rms 10000 (~-10 dBFS) |
|---|---|---|---|---|
| semua 1/2 (1/512) | 34,3 ; 0 | 69,1 ; 0 | 77,3 ; 0 | 83,7 ; 0 |
| 1 stage tanpa skala | 40,3 ; 0 | 74,9 ; 0 | 81,6 ; 0 | 84,9 ; 0 |
| 2 stage tanpa skala | 46,3 ; 0 | 79,8 ; 0 | 84,0 ; 0 | 85,2 ; 0 |
| 3 stage tanpa skala | 52,5 ; 0 | 83,1 ; 0 | 84,9 ; 0 | 33,0 ; **12** |

## Fitur int8 (W=24)
Persentil 0,5 / 99,5 dari L (Q6.8): 3380 / 9999; maks 11.509; 0,01% elemen nol. FEAT_SHIFT = 5 menutupi
(9999-3380)/32 = 207 < 255 langkah. FEAT_OFFSET = 6690 (titik tengah).

## Detektor (stream sintetis 20 s x 12 untuk recall, 60 s x 12 untuk negatif; frame onset ±3..+14 dianggap tepat)
| K | S | recall 6 dB | 10 dB | 15 dB | 25 dB | FP derau | klik | lompatan x4 | nada keras |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 6 | 1,00 | 1,00 | 1,00 | 1,00 | 0,00 | 0,00 | 1,00 | 6,7 |
| 6 | 6 | 1,00 | 1,00 | 1,00 | 1,00 | 0,00 | 0,00 | 1,00 | 5,0 |
| **8** | **6** | 0,61 | 1,00 | 1,00 | 1,00 | 0,00 | 0,00 | 1,00 | 4,7 |
| 12 | 6 | 0,22 | 0,94 | 1,00 | 1,00 | 0,00 | 0,00 | 0,33 | 3,0 |
| 16 | 6 | 0,06 | 0,89 | 0,94 | 1,00 | 0,00 | 0,00 | 0,00 | 1,7 |

(FP per menit; S = 6 tetap.) K=4 dan K=6 memberi recall lebih tinggi pada 6 dB dengan FP yang sama pada
negatif sintetis. K=8 tetap dipilih karena aturan seleksi (gerbang recall 10 dB, lalu terdekat dengan nilai awal)
ditetapkan sebelum melihat hasil, dan karena negatif sintetis tidak mencakup latar nyata (HVAC, bicara, derau
yang naik perlahan) yang justru paling menguji margin K. Keputusan ini layak ditinjau ulang pada data nyata.
Tabel penuh ada di JSON mentah.
