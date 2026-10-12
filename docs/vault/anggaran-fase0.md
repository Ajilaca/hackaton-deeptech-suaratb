# Anggaran Fase 0 (dihasilkan oleh `python -m golden.analysis.budget`)

> Hitungan ANALITIK dari parameter. Bukan hasil sintesis. Kapasitas perangkat: dari ingatan, verifikasi di datasheet.

### Konfigurasi P=16, FFT_W=24, kanal=[16, 32, 32], FC=16
Bobot: 14496  MAC/inferensi: 3,698,448  (analitik)

| Memori | kata x lebar | bit |
|---|---|---|
| weight_rom (ROM) | 921 x 128 | 117,888 |
| bias_rom (ROM) | 97 x 24 | 2,328 |
| requant_rom (ROM) | 5 x 21 | 105 |
| act_buf_A (RAM) | 3840 x 8 | 30,720 |
| act_buf_B (RAM) | 8000 x 8 | 64,000 |
| feat_buf0 (RAM) | 2000 x 8 | 16,000 |
| sample_ring (RAM) | 512 x 16 | 8,192 |
| fft_ram (RAM) | 512 x 48 | 24,576 |
| twiddle_rom (ROM) | 256 x 32 | 8,192 |
| window_rom (ROM) | 200 x 16 | 3,200 |
| mel_rom (ROM) | 493 x 19 | 9,367 |
| mel_acc (RAM) | 40 x 58 | 2,320 |
| log_lut (ROM) | 256 x 8 | 2,048 |
| **total (logis)** | | **288,936** (282 Kbit) |

Siklus CNN ideal: conv1=18,000, conv2=144,000, conv3=69,120, fc1=32, fc2=16, total=231,168
Latensi CNN ideal @100 MHz: 2.31 ms; @150 MHz: 1.54 ms (target/estimasi, belum diukur)

Kapasitas perangkat: dari ingatan, verifikasi di datasheet.

| Perangkat | memori logis / kapasitas | blok | pengali / tersedia | LUT (est. 4-8 rb) | memori | pengali | muat? |
|---|---|---|---|---|---|---|---|
| XC7A35T | 288,936 / 1,843,200 (16%) | 17/50 | 31/90 | ya | ok | ok | ya (estimasi) |
| EP4CE22 | 288,936 / 608,256 (48%) | 38/66 | 42/66 | ya | ok | ok | ya (estimasi) |
| EP4CE6 | 288,936 / 276,480 (105%) | 38/30 | 42/15 | mungkin | LEWAT | LEWAT | TIDAK |
| ECP5-25F | 288,936 / 1,032,192 (28%) | 24/56 | 42/56 | ya | ok | ok | ya (estimasi) |

### Konfigurasi P=8, FFT_W=24, kanal=[16, 32, 32], FC=16
Bobot: 14496  MAC/inferensi: 3,698,448  (analitik)

| Memori | kata x lebar | bit |
|---|---|---|
| weight_rom (ROM) | 1826 x 64 | 116,864 |
| bias_rom (ROM) | 97 x 24 | 2,328 |
| requant_rom (ROM) | 5 x 21 | 105 |
| act_buf_A (RAM) | 3840 x 8 | 30,720 |
| act_buf_B (RAM) | 8000 x 8 | 64,000 |
| feat_buf0 (RAM) | 2000 x 8 | 16,000 |
| sample_ring (RAM) | 512 x 16 | 8,192 |
| fft_ram (RAM) | 512 x 48 | 24,576 |
| twiddle_rom (ROM) | 256 x 32 | 8,192 |
| window_rom (ROM) | 200 x 16 | 3,200 |
| mel_rom (ROM) | 493 x 19 | 9,367 |
| mel_acc (RAM) | 40 x 58 | 2,320 |
| log_lut (ROM) | 256 x 8 | 2,048 |
| **total (logis)** | | **287,912** (281 Kbit) |

Siklus CNN ideal: conv1=36,000, conv2=288,000, conv3=138,240, fc1=64, fc2=16, total=462,320
Latensi CNN ideal @100 MHz: 4.62 ms; @150 MHz: 3.08 ms (target/estimasi, belum diukur)

Kapasitas perangkat: dari ingatan, verifikasi di datasheet.

| Perangkat | memori logis / kapasitas | blok | pengali / tersedia | LUT (est. 4-8 rb) | memori | pengali | muat? |
|---|---|---|---|---|---|---|---|
| XC7A35T | 287,912 / 1,843,200 (16%) | 17/50 | 23/90 | ya | ok | ok | ya (estimasi) |
| EP4CE22 | 287,912 / 608,256 (47%) | 38/66 | 34/66 | ya | ok | ok | ya (estimasi) |
| EP4CE6 | 287,912 / 276,480 (104%) | 38/30 | 34/15 | mungkin | LEWAT | LEWAT | TIDAK |
| ECP5-25F | 287,912 / 1,032,192 (28%) | 24/56 | 34/56 | ya | ok | ok | ya (estimasi) |

### Konfigurasi P=32, FFT_W=24, kanal=[16, 32, 32], FC=16
Bobot: 14496  MAC/inferensi: 3,698,448  (analitik)

| Memori | kata x lebar | bit |
|---|---|---|
| weight_rom (ROM) | 489 x 256 | 125,184 |
| bias_rom (ROM) | 97 x 24 | 2,328 |
| requant_rom (ROM) | 5 x 21 | 105 |
| act_buf_A (RAM) | 3840 x 8 | 30,720 |
| act_buf_B (RAM) | 8000 x 8 | 64,000 |
| feat_buf0 (RAM) | 2000 x 8 | 16,000 |
| sample_ring (RAM) | 512 x 16 | 8,192 |
| fft_ram (RAM) | 512 x 48 | 24,576 |
| twiddle_rom (ROM) | 256 x 32 | 8,192 |
| window_rom (ROM) | 200 x 16 | 3,200 |
| mel_rom (ROM) | 493 x 19 | 9,367 |
| mel_acc (RAM) | 40 x 58 | 2,320 |
| log_lut (ROM) | 256 x 8 | 2,048 |
| **total (logis)** | | **296,232** (289 Kbit) |

Siklus CNN ideal: conv1=18,000, conv2=72,000, conv3=34,560, fc1=32, fc2=16, total=124,608
Latensi CNN ideal @100 MHz: 1.25 ms; @150 MHz: 0.83 ms (target/estimasi, belum diukur)

Kapasitas perangkat: dari ingatan, verifikasi di datasheet.

| Perangkat | memori logis / kapasitas | blok | pengali / tersedia | LUT (est. 4-8 rb) | memori | pengali | muat? |
|---|---|---|---|---|---|---|---|
| XC7A35T | 296,232 / 1,843,200 (16%) | 17/50 | 47/90 | ya | ok | ok | ya (estimasi) |
| EP4CE22 | 296,232 / 608,256 (49%) | 39/66 | 58/66 | ya | ok | ok | ya (estimasi) |
| EP4CE6 | 296,232 / 276,480 (107%) | 39/30 | 58/15 | mungkin | LEWAT | LEWAT | TIDAK |
| ECP5-25F | 296,232 / 1,032,192 (29%) | 24/56 | 58/56 | ya | ok | LEWAT | TIDAK |

### Konfigurasi P=8, FFT_W=24, kanal=[8, 16, 16], FC=16
Bobot: 3800  MAC/inferensi: 996,752  (analitik)

| Memori | kata x lebar | bit |
|---|---|---|
| weight_rom (ROM) | 489 x 64 | 31,296 |
| bias_rom (ROM) | 57 x 24 | 1,368 |
| requant_rom (ROM) | 5 x 21 | 105 |
| act_buf_A (RAM) | 2000 x 8 | 16,000 |
| act_buf_B (RAM) | 4000 x 8 | 32,000 |
| feat_buf0 (RAM) | 2000 x 8 | 16,000 |
| sample_ring (RAM) | 512 x 16 | 8,192 |
| fft_ram (RAM) | 512 x 48 | 24,576 |
| twiddle_rom (ROM) | 256 x 32 | 8,192 |
| window_rom (ROM) | 200 x 16 | 3,200 |
| mel_rom (ROM) | 493 x 19 | 9,367 |
| mel_acc (RAM) | 40 x 58 | 2,320 |
| log_lut (ROM) | 256 x 8 | 2,048 |
| **total (logis)** | | **154,664** (151 Kbit) |

Siklus CNN ideal: conv1=18,000, conv2=72,000, conv3=34,560, fc1=32, fc2=16, total=124,608
Latensi CNN ideal @100 MHz: 1.25 ms; @150 MHz: 0.83 ms (target/estimasi, belum diukur)

Kapasitas perangkat: dari ingatan, verifikasi di datasheet.

| Perangkat | memori logis / kapasitas | blok | pengali / tersedia | LUT (est. 4-8 rb) | memori | pengali | muat? |
|---|---|---|---|---|---|---|---|
| XC7A35T | 154,664 / 1,843,200 (8%) | 13/50 | 23/90 | ya | ok | ok | ya (estimasi) |
| EP4CE22 | 154,664 / 608,256 (25%) | 24/66 | 34/66 | ya | ok | ok | ya (estimasi) |
| EP4CE6 | 154,664 / 276,480 (56%) | 24/30 | 34/15 | mungkin | ok | LEWAT | TIDAK |
| ECP5-25F | 154,664 / 1,032,192 (15%) | 16/56 | 34/56 | ya | ok | ok | ya (estimasi) |

### Konfigurasi P=8, FFT_W=18, kanal=[8, 16, 16], FC=16
Bobot: 3800  MAC/inferensi: 996,752  (analitik)

| Memori | kata x lebar | bit |
|---|---|---|
| weight_rom (ROM) | 489 x 64 | 31,296 |
| bias_rom (ROM) | 57 x 24 | 1,368 |
| requant_rom (ROM) | 5 x 21 | 105 |
| act_buf_A (RAM) | 2000 x 8 | 16,000 |
| act_buf_B (RAM) | 4000 x 8 | 32,000 |
| feat_buf0 (RAM) | 2000 x 8 | 16,000 |
| sample_ring (RAM) | 512 x 16 | 8,192 |
| fft_ram (RAM) | 512 x 36 | 18,432 |
| twiddle_rom (ROM) | 256 x 32 | 8,192 |
| window_rom (ROM) | 200 x 16 | 3,200 |
| mel_rom (ROM) | 493 x 19 | 9,367 |
| mel_acc (RAM) | 40 x 58 | 2,320 |
| log_lut (ROM) | 256 x 8 | 2,048 |
| **total (logis)** | | **148,520** (145 Kbit) |

Siklus CNN ideal: conv1=18,000, conv2=72,000, conv3=34,560, fc1=32, fc2=16, total=124,608
Latensi CNN ideal @100 MHz: 1.25 ms; @150 MHz: 0.83 ms (target/estimasi, belum diukur)

Kapasitas perangkat: dari ingatan, verifikasi di datasheet.

| Perangkat | memori logis / kapasitas | blok | pengali / tersedia | LUT (est. 4-8 rb) | memori | pengali | muat? |
|---|---|---|---|---|---|---|---|
| XC7A35T | 148,520 / 1,843,200 (8%) | 13/50 | 21/90 | ya | ok | ok | ya (estimasi) |
| EP4CE22 | 148,520 / 608,256 (24%) | 23/66 | 24/66 | ya | ok | ok | ya (estimasi) |
| EP4CE6 | 148,520 / 276,480 (54%) | 23/30 | 24/15 | mungkin | ok | LEWAT | TIDAK |
| ECP5-25F | 148,520 / 1,032,192 (14%) | 15/56 | 24/56 | ya | ok | ok | ya (estimasi) |

## Catatan metode
- Memori = bit logis (kata x lebar); blok dihitung per memori, minimal 1 blok per memori, tanpa memperhitungkan aspek rasio/porting ganda. Bit nyata di sintesis bisa lebih besar.
- Pengali: hanya blok pengali keras (DSP48E1 25x18 / 18x18) dengan pemecahan operan lebar; pengali 8x8 PE dihitung 1 blok/PE. Mode 9x9 ganda (Cyclone IV, ECP5) atau pengali di LUT tidak dihitung, sehingga EP4CE6 'LEWAT' pada pengali bisa dilonggarkan dengan memindahkan sebagian PE ke LUT.
- LUT: tidak dihitung dari parameter; memakai estimasi kasar 4-8 rb dari estimasi awal proyek (keyakinan rendah, faktor 2).
- feat_buf dihitung terpisah dari act_buf_A; menyatukannya (alias) menghemat 16 Kbit.
- Siklus: ideal output-stationary, tanpa overhead (~+10%); latensi @100/150 MHz adalah TARGET, bukan Fmax terukur.
