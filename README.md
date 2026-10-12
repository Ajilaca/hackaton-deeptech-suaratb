# SuaraTB

Suara batuk berpotensi menjadi salah satu sumber informasi untuk skrining tuberkulosis (TBC). Mengenali pola dari audio batuk saja
belum cukup: modelnya harus diuji pada data yang sesuai, dan implementasinya harus bisa berjalan di perangkat dengan sumber daya
terbatas.

SuaraTB mengeksplorasi pendekatan itu lewat akselerator FPGA. Audio dari mikrofon I2S diolah menjadi fitur log-mel, lalu CNN int8
menghitung skor risiko. Seluruh proses berlangsung di perangkat, sehingga audio tidak perlu dikirim ke server. Hardware ditulis dalam
SystemVerilog, dan model numpy di `golden/` menjadi acuan untuk memeriksa bahwa RTL menghasilkan angka yang persis sama.

Proyek ini dibuat untuk Hackathon Deeptech 2026 (Indonesia Chip Hackathon Challenge, ITB/ICDeC) dengan tema *Medical Device*.

## Cara kerja

Mikrofon I2S mengirim sampel 16 kHz ke `i2s_rx`. Sampel melewati pre-emphasis, dipotong menjadi frame yang saling tumpang tindih,
diberi jendela Hamming, lalu diubah ke spektrum dengan FFT 512 titik dan dipetakan ke 40 band mel. Logaritma diambil dengan tabel
dan hasilnya dikuantisasi ke int8, sehingga satu batuk menjadi segmen fitur berukuran 40x50. Detektor batuk membandingkan energi
tiap hop dengan derau latar yang dilacak (EMA) dan menentukan bagian audio yang diambil.

Segmen itu masuk ke CNN: tiga lapis konvolusi 3x3 dengan ReLU dan MaxPool, lalu GAP dan dua lapis FC yang menghasilkan satu logit.
Akselerator memakai skema output-stationary dengan P lane paralel, dan logitnya dikirim lewat UART. Di sisi perangkat lunak host,
logit dari beberapa batuk dirata-rata dan dapat digabung dengan gejala klinis (`sw/`, belum berisi kode).

![Arsitektur SuaraTB: mikrofon I2S ke skor risiko](docs/img/arsitektur.png)

Mode mikrofon memakai detektor batuk untuk memilih segmen; mode dataset memproses 50 frame pertama klip yang dikirim lewat UART
(`rtl/sys/suaratb_top.sv`).

Dua keputusan desain yang memengaruhi seluruh repo: setiap blok punya padanan numpy bit-exact di `golden/` yang berfungsi sebagai
spesifikasi, dan RTL dibuat vendor-neutral (tanpa IP atau primitif vendor; kode spesifik perangkat hanya di `target/`). Semua ukuran
(P, lebar data, jumlah kanal) berasal dari `golden/params.py`, sehingga konfigurasi dapat diperkecil untuk FPGA yang lebih kecil.

| Bagian | Nilai |
|---|---|
| Audio | 16 kHz, 16 bit, mikrofon I2S (INMP441), klip 0,5 s |
| Front-end | Pre-emphasis 0,97, frame 400/hop 160, Hamming, FFT 512 radix-2 (data 24 bit), mel 40 band, log2 LUT, int8 |
| Detektor batuk | Energi per hop, EMA derau, ambang 8x, tahan 300 ms |
| CNN | Conv3x3 16/32/32 + ReLU + MaxPool, GAP, FC 32->16->1; 14.496 bobot int8, 3.698.448 MAC |
| Akselerator | Output-stationary, P lane paralel (8/16/32), akumulator 24 bit, requant per layer |
| Antarmuka | I2S masuk, UART keluar (skor) dan mode dataset lewat UART, CSR 8 bit |

## Data

Data audio berasal dari dataset publik Kaggle "Google + CIDRZ Health AI Evaluation Zambia": rekaman batuk di tiga klinik Lusaka
dengan satu perekam dan tiga ponsel, serta metadata klinis 1.828 peserta (205 positif TB). Audio yang dapat dipasangkan dengan label
mencakup 364 peserta (60 positif). Pembagian data selalu per peserta, bukan per klip. Data, audio, fitur, bobot, dan proyek Vivado
disimpan di luar repo (`$SUARATB_DATA`, awal `/data/suaratb`).

## Hasil dan evaluasi

Ada dua pertanyaan yang berbeda di sini: apakah hardware bekerja sesuai model referensinya, dan apakah modelnya cukup baik untuk
membantu skrining. Jawaban pertama positif, jawaban kedua belum. Keberhasilan implementasi hardware tidak otomatis berarti modelnya
memadai secara klinis.

**Hardware.** RTL menghasilkan keluaran yang sama persis dengan golden pada tes per modul dan ujung ke ujung (mode dataset dan
mikrofon), untuk P = 8, 16, dan 32, termasuk dengan bobot hasil distilasi. Regresi terpadu 20 tes RTL dan 67 tes golden lulus di
simulator Icarus. Sintesis Vivado untuk Artix-7 XC7A35T-1 memenuhi constraint 6,667 ns (150 MHz) untuk ketiga nilai P, tetapi hanya
dalam mode out-of-context, satu kali jalan per P tanpa seed sweep, dan margin untuk P = 32 hanya 0,023 ns. Clock inti yang dipakai di
`target/` adalah 147,456 MHz agar BCLK I2S dan baud UART pas. Semua angka ini berasal dari simulasi dan laporan sintesis.

| Aspek | Hasil | Sumber |
|---|---|---|
| Kesesuaian RTL dengan golden | Bit-exact per modul dan ujung ke ujung (dataset dan mikrofon), P = 8/16/32 | `docs/vault/verifikasi.md` |
| Timing Vivado XC7A35T-1 (OOC) | 100 MHz terpenuhi; 150 MHz terpenuhi untuk P = 8/16/32 (WNS +0,023 s.d. +0,222 ns) | `docs/vault/sintesis.md` |
| Sumber daya, P = 16 @ 150 MHz | 5.926 LUT (28,5%), 7.799 FF, 17 DSP48E1, 12 tile BRAM, 0,19 W (vectorless) | `docs/vault/sintesis.md` |

**Model.** Pembanding terbaik adalah model gejala klinis (AUROC 0,77), bukan audio. CNN yang dilatih dari nol pada audio hanya
mencapai AUROC sekitar 0,56 dan tidak dapat dibedakan dari hasil acak. Embedding pralatih HeAR memberi sinyal yang lebih jelas:
setelah jenis kelamin dikendalikan, AUROC-nya 0,71 dan lolos uji permutasi. Namun tambahan di atas model klinis hanya sekitar +0,04
dengan interval kepercayaan yang mencakup nol, sehingga manfaat audio di atas gejala klinis belum terbukti, dan hasil lintas klinik
belum konsisten. HeAR dipakai sebagai referensi di sisi host, bukan di FPGA. Untuk membawa sebagian sinyalnya ke hardware, embedding
itu didistilasi ke CNN yang arsitekturnya sama dengan golden: AUROC 0,654 (guru 0,672), turun 0,008 setelah kuantisasi int8, dan
bit-exact di RTL. Hasil distilasi ini eksploratori.

| Aspek | Hasil | Sumber |
|---|---|---|
| Gejala klinis saja | AUROC 0,77 (1.828 peserta, 205 positif) | `docs/vault/model.md` §5b |
| Audio, CNN dilatih dari nol | AUROC ~0,56; tak dapat dibedakan dari nol permutasi | `docs/vault/model.md` §4 |
| Audio, embedding HeAR (364 peserta, 60 positif) | AUROC 0,71 dalam jenis kelamin, p disesuaikan 0,002; tambahan di atas klinis +0,04 dengan CI mencakup 0 | `docs/vault/model.md` §5d |
| Distilasi HeAR -> CNN golden | AUROC 0,654 (guru 0,672); RTL bit-exact dengan bobot ini | `docs/vault/model.md` §5e |
| Kuantisasi int8 | Penurunan AUROC 0,001 (model dari nol), 0,008 (murid distilasi); kriteria <= 0,02 | `docs/vault/model.md` |
| Standar WHO TPP | Tidak dipenuhi | `docs/vault/model.md` §5c |

Laporan lengkap ada di `docs/result/result.md` (dan `result.pdf`).

## Keterbatasan

- SuaraTB dimaksudkan untuk mendukung triase dan prioritas rujukan ke pemeriksaan lanjutan, bukan untuk mendiagnosis TBC.
- Target WHO untuk triase (sensitivitas minimal 90% dan spesifisitas minimal 70%) belum terpenuhi.
- Desain belum diuji di papan FPGA. Jenis papan di lokasi acara belum diumumkan (Fase 7).
- Evaluasi audio memakai satu dataset dari tiga klinik dengan 364 peserta berlabel (60 positif); tidak ada klaim generalisasi.
- Syarat tambahan dataset (di luar ODbL-1.0 dari metadata Kaggle) belum dibaca, dan bobot model tidak ada di repo.

## Status

Fase 0 sampai 6 selesai, termasuk versi 150 MHz (cabang `optimized`) dan remodeling model. Fase T (regresi terpadu) lulus di
simulasi. Sisanya: papan FPGA (Fase 7, bila diumumkan) dan benchmark CPU (Fase 8). Rincian di `docs/vault/status-fase.md`.

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

Setiap direktori punya `README.md` sendiri.

## Dokumen

Baca berurutan: `docs/vault/status-fase.md`, `docs/vault/spesifikasi.md`, `docs/vault/keputusan.md`. Indeks: `docs/CONTEXT_INDEX.md`.
Perintah build dan tes ada di `Makefile` (`make help`) dan docstring skrip; README tidak mengulangnya.

## Referensi

- Dataset: [Google + CIDRZ Health AI Evaluation Zambia di Kaggle](https://www.kaggle.com/datasets/googlehealthai/google-health-ai)
- Model embedding akustik: [HeAR (`google/hear-pytorch`) di Hugging Face](https://huggingface.co/google/hear-pytorch)
- Pembanding embedding: OPERA (OPERA-CE dan OPERA-CT); tautan belum dicantumkan karena belum diverifikasi.

## Lisensi

Lihat `LICENSE`. Data Zambia CIDRZ (ODbL-1.0) dan TBscreen (CC BY 4.0) tidak didistribusikan di repo ini.
