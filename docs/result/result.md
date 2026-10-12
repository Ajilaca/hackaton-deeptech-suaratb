# SuaraTB: laporan hasil

Akselerator skrining TBC dari suara batuk (Hackathon Deeptech 2026, Indonesia Chip Hackathon Challenge, tema Medical Device).
Laporan ini merangkum semua yang dibangun dan dijalankan, hasil terukurnya, dan batasannya. Angka berasal dari skrip di repo,
laporan alat (Vivado 2026.1, Yosys 0.33), dan simulasi (Icarus 12 + cocotb 2.1), per 12 Oktober 2026.

## 1. Ringkasan

| Pertanyaan | Jawaban | Bukti |
|---|---|---|
| Apakah pipeline audio -> skor berjalan di akselerator khusus? | **Ya**, bit-exact terhadap model numpy (golden) pada tiap modul dan ujung ke ujung (mode dataset dan mikrofon) | §4 |
| Apakah muat dan cukup cepat di FPGA kelas kecil? | **Ya**: XC7A35T-1, 150 MHz terpenuhi untuk P = 8/16/32; P = 16 memakai 5.926 LUT (28,5%), 17 DSP, 12 tile BRAM | §5 |
| Apakah gejala klinis memberi skrining yang berguna? | AUROC 0,77 pada 1.828 peserta; **belum** memenuhi standar WHO TPP | §7.2 |
| Apakah audio batuk mendeteksi TBC? | CNN dari nol: tidak (AUROC ~0,56). Embedding pralatih (HeAR): ada indikasi sinyal (AUROC 0,71 dalam jenis kelamin, p = 0,002), tambahan di atas gejala klinis kecil dan **belum terbukti** | §7.3-7.5 |
| Apakah sinyal itu bisa dipindahkan ke hardware? | Sebagian besar: murid distilasi AUROC 0,654 (guru 0,672), kuantisasi int8 turun 0,008, RTL bit-exact | §7.6 |

**Posisi klaim.** Alat adalah pendukung triase/prioritas rujukan ke tes TCM/dahak, bukan diagnosis. Klaim yang didukung bukti:
pipeline lengkap audio -> skor di akselerator khusus dengan latensi dan sumber daya terukur, bobot dapat diganti tanpa mengubah
hardware. Klaim yang **tidak** didukung: akurasi diagnosis, generalisasi lintas populasi, memenuhi standar WHO, "jalan di papan".

## 2. Sistem

| Bagian | Isi |
|---|---|
| Masukan | Mikrofon I2S (INMP441), 16 kHz, 16 bit; atau klip dataset lewat UART |
| Front-end | Pre-emphasis 0,97, frame 400/hop 160, Hamming, FFT 512 radix-2 (data 24 bit), mel 40 band, log2 LUT, kuantisasi int8 |
| Detektor | Energi per hop, EMA derau, ambang 8x selama >= 2 hop, tahan 300 ms; segmen 50 frame x 40 mel (5 pre-roll) |
| CNN | Conv3x3 16/32/32 + ReLU + MaxPool, GAP, FC 32->16->1; 14.496 bobot int8, 3.698.448 MAC; akumulator 24 bit |
| Akselerator | Output-stationary, P lane paralel (P = 8/16/32), requant per layer (M0 Q15, shift, saturasi) |
| Keluaran | Logit int16 per batuk lewat UART; skor pasien = rata-rata 3-5 batuk di host; fusi klinis di host |

Prinsip rekayasa: golden numpy bit-exact adalah spesifikasi; RTL vendor-neutral (tanpa IP vendor di `rtl/`); semua ukuran adalah
parameter dari satu sumber (`golden/params.py`); angka hardware hanya dari laporan alat.

## 3. Fase 0: parameter dan kalibrasi

Parameter dibekukan setelah eksperimen pada audio sintetis lalu audio nyata (perekam Zambia, 1.764 klip dari 150 rekaman).

| Keputusan | Hasil | Alasan |
|---|---|---|
| Lebar FFT = 24 bit (nilai awal 18) | Galat fitur 0,007 oktaf; W = 18 gagal (0,79 oktaf), W = 22 lolos (0,029) | Ambang 0,0625 oktaf (0,5 langkah int8) |
| Offset fitur 8265 (sintetis: 6690) | Jenuh int8 0,2% (sebelumnya 2,0%) | K-14, dari audio nyata |
| Skala FFT 1/2 di semua stage | 0 sampel jenuh pada 60 klip berpuncak tertinggi | K-04 |
| Akumulator CNN 24 bit | Puncak teramati 258 rb (batas 8,4 juta) | K-10 |
| Detektor K = 8, EMA 2^-6 | ±15 kejadian/menit pada rekaman utuh; tanpa ground truth batuk, presisi tidak terukur | K-07 |

Tes golden: 67 lulus (blok, CNN, jadwal FFT, pengalamatan DSP, front-end, parameter, tabel). Anggaran analitik: memori logis
282 Kbit (P = 16), 3,7 juta MAC; muat di XC7A35T, EP4CE22, ECP5-25F; tidak muat di EP4CE6.

## 4. Verifikasi RTL

Metode: vektor sintetis dari golden dibandingkan bit-exact (`==` integer, tanpa toleransi) dengan keluaran RTL di Icarus + cocotb.

| Cakupan | Hasil |
|---|---|
| Front-end: `i2s_rx`, `preemph`, `frame_buf`, `cough_det` | Lulus |
| DSP: `window`, `fft512`, `power`, `mel`, `log_q`, `quant`, `feat_buf`, `dsp_chain` | Lulus (12 modul Fase 2-3) |
| CNN: `requant`, `gap`, `cnn_top` | Lulus: bobot acak (5 kasus), bobot terlatih (8 klip nyata), P = 8/16/32 |
| Sistem: `suaratb_top` mode dataset (UART, 2 kasus) dan mikrofon (I2S -> skor) | Lulus ujung ke ujung |
| Setelah perbaikan timing (100 MHz) dan pipeline 150 MHz | Semua tes di atas diulang dan lulus; `test_top` lulus dua kali |
| Dengan bobot murid distilasi | `cnn_top`, mode mikrofon, dan dataset + mikrofon lulus bit-exact |

Dua cacat RTL ditemukan oleh tes ujung ke ujung dan diperbaiki: DSP tidak dibersihkan saat ganti mode, dan segmen CNN dibaca
sebelum frame ke-50 selesai. Simulasi RTL hanya di Icarus; Verilator dipakai untuk lint.

## 5. Sintesis (Vivado XC7A35T-1, mode OOC, setelah route)

Tanpa constraint I/O dan tanpa MMCM (tidak ada pin); timing = jalur register ke register dalam inti. Model terlatih `model_v2`.

| P | Clock | LUT | FF | DSP48E1 | BRAM tile | WNS | Daya* |
|---|---|---|---|---|---|---|---|
| 8 | 100 MHz | 4.823 (23,2%) | 5.548 | 17 | 12 | +0,443 ns | 0,138 W |
| 16 | 100 MHz | 5.834 (28,1%) | 6.148 | 17 | 12 | +0,077 ns | 0,149 W |
| 32 | 100 MHz | 8.236 (39,6%) | 7.486 | 17 | 12 | +0,258 ns | 0,182 W |
| 8 | 150 MHz | 4.807 (23,1%) | 6.736 | 17 | 12 | +0,111 ns | 0,168 W |
| **16** | **150 MHz** | **5.926 (28,5%)** | **7.799** | 17 | 12 | **+0,222 ns** | 0,192 W |
| 32 | 150 MHz | 8.120 (39,0%) | 10.071 | 17 | 12 | +0,023 ns | 0,246 W |

\* Estimasi vectorless (keyakinan "Medium"), bukan pengukuran.

- Jalur kritis berpindah dari kupu-kupu FFT, ke detektor, lalu ke baca BRAM + pengali PE; semuanya diperbaiki tanpa mengubah
  hasil bit-exact (register keluaran BRAM, pengali dipecah, lebar aritmetika disempitkan).
- DSP = 17 untuk semua P: pengali 8x8 PE dipetakan ke LUT; kenaikan P menambah LUT, bukan DSP. BRAM tidak bergantung P.
- Siklus inferensi CNN (simulasi, versi 100 MHz): P = 8: 451.851; P = 16: 226.490; P = 32: 122.843 (belum diukur ulang pada versi 150 MHz).
- Pembanding Yosys 0,33 (tanpa timing): LUT 11.083-12.560, DSP 32-56; bukan prediksi Vivado.
- Latensi yang dirasakan pengguna didominasi pengambilan 0,5 s audio; komputasi CNN hanya ±1-2 ms.
- Batas: satu jalankan per titik (tanpa seed sweep), margin P = 32 tipis (+0,023 ns), belum diuji di papan.

## 6. Data

| Sumber | Peserta | Positif | Catatan |
|---|---|---|---|
| Zambia CIDRZ (Kaggle, ODbL-1.0), metadata | 1.828 | 205 | 3 klinik Lusaka; label `ground_truth_tb` (Xpert/kultur) |
| Zambia, dengan audio | 364 | 60 | 4 perangkat serentak (perekam, Phone A/B/C); rilis Kaggle memuat 1.481 berkas |
| TBscreen (Nairobi, CC BY 4.0) | 123 dengan audio | 87 | Cohort ≈ label (perancu rekrutmen) |
| CODA TB DREAM | - | - | Tidak dipakai: butuh akses terverifikasi (K-15) |

Audit (E1): angka "664 peserta" pada deskripsi dataset tidak ada di rilis; pemulihan barcode salah ketik tidak menambah peserta.
Split selalu per peserta. Hanya dataset Zambia yang dipakai untuk pelatihan; tidak ada data dari INMP441.

## 7. Model dan hasil klinis

Metrik: AUROC dengan CI 95% bootstrap atas peserta; nol permutasi (label diacak dalam klinik/jenis kelamin dan seluruh pipeline
diulang); kontrol jenis kelamin dan usia; uji lintas perangkat/dataset/klinik. Hasil dilaporkan apa adanya, termasuk yang negatif.

### 7.1 Kuantisasi

| Model | AUROC float | AUROC int8 golden | Selisih | Kriteria <= 0,02 |
|---|---|---|---|---|
| CNN dari nol (`model_v2`) | 0,545 | 0,546 | +0,001 | lulus |
| Murid distilasi HeAR | 0,653 | 0,645 | 0,008 | lulus |

Angka kuantisasi lulus tidak membuktikan kemampuan klinis; logit `model_v2` hampir konstan.

### 7.2 Gejala klinis dan standar WHO

| Model | n / positif | AUROC | CI 95% |
|---|---|---|---|
| Gejala klinis 14 fitur (LR), semua peserta | 1.828 / 205 | 0,772 | 0,74-0,81 |
| Klinis diperluas | 1.828 / 205 | 0,772 | 0,74-0,81 |
| Jenis kelamin + usia saja (360 peserta) | 360 / 58 | 0,654 | 0,58-0,72 |

| Syarat WHO TPP triase (sens >= 90%, spes >= 70%) | Hasil skor klinis |
|---|---|
| Sensitivitas >= 90% | spesifisitas 38% (butuh 70%) |
| Spesifisitas >= 70% | sensitivitas 71% (butuh 90%) |

Memenuhi 90%/70% sekaligus butuh AUROC ≈ 0,90 (aproksimasi binormal). **Tidak dipenuhi dan tidak diklaim.**

### 7.3 Audio, CNN dilatih dari nol (360 peserta, 58 positif)

| Model | AUROC | Catatan |
|---|---|---|
| CNN MIL | 0,562 (CI 0,48-0,64) | nol 0,523 (p95 0,591), p-empiris 0,41 |
| CNN diperbaiki (derau, klip mirip-batuk) | 0,45-0,49 | di bawah acak |
| LR statistik log-mel | 0,66-0,70 mentah; 0,58-0,60 setelah jenis kelamin+usia | sebagian besar jenis kelamin |
| Audio memprediksi jenis kelamin | 0,89 | prevalensi TB laki-laki 25% vs perempuan 8% |
| VGGish pralatih | 0,51 (uji tidak sah: jendela 47% nol) | tidak disimpulkan |
| Lintas dataset Zambia <-> TBscreen | 0,50-0,55 | tidak berpindah |
| Lintas perangkat (log-mel) | 0,52-0,63 | tidak lebih baik dari dalam-perangkat |

### 7.4 Embedding pralatih dengan protokol tetap (364 peserta, 60 positif)

Protokol (E0): CV 5 x 5-10 per peserta, stratifikasi label x klinik x jenis kelamin; skor klinis dilatih hanya pada peserta tanpa
audio; metrik utama AUROC terstratifikasi jenis kelamin dan Δ fusi - klinis; nol permutasi dalam klinik x jenis kelamin.
Semua kandidat dilaporkan.

| Fitur | AUROC strat. jenis kelamin | Nol p95 / maks | Δ vs klinis (CI 95%) |
|---|---|---|---|
| Statistik log-mel (pembanding) | 0,599 | 0,619 / 0,672 | -0,002 (-0,032..+0,029) |
| HeAR | 0,675 | 0,561 / 0,582 | +0,015 (-0,023..+0,058) |
| OPERA-CT | 0,677 | 0,609 / 0,644 | +0,029 (-0,011..+0,068) |
| OPERA-CE | 0,606 | 0,604 / 0,612 | +0,013 (-0,013..+0,041) |

### 7.5 Uji permutasi final dan pemeriksaan

1.000 permutasi, statistik maksimum atas tiga kandidat, permutasi sama untuk semua kandidat, C tetap 0,03 (ditetapkan di muka;
hasil serupa untuk C = 0,01 dan 0,1). Kriteria lolos distilasi ditetapkan sebelum angka final.

| Kandidat | AUROC strat. | p disesuaikan (label) | Δ di atas klinis | p (Δ) | CI bootstrap Δ |
|---|---|---|---|---|---|
| HeAR | 0,709 | 0,002 | +0,039 | 0,003 | -0,011..+0,092 |
| OPERA-CT | 0,664 | 0,006 | +0,025 | 0,015 | -0,017..+0,064 |
| OPERA-CE | 0,621 | 0,089 | +0,022 | 0,023 | -0,003..+0,048 |

| Pemeriksaan | Hasil |
|---|---|
| Kontrol negatif (fitur Gaussian acak, 2 seed) | Bersih: tidak ada p disesuaikan mendekati 0,05 |
| Lintas perangkat HeAR (Phone A/B/C, peserta sama) | AUROC strat. 0,62-0,69 di 9 pasangan; ketahanan terhadap mikrofon, bukan populasi |
| Jalan pintas HeAR | Embedding memprediksi jenis kelamin 0,95 dan klinik Kan 0,90; AUROC dalam Kan 0,73, Cha 0,64 (11 positif) |
| Leave-one-clinic-out | Kan->Cha: HeAR 0,55, OPERA-CT 0,73; Cha->Kan: HeAR 0,62, OPERA-CT 0,56. Tidak konsisten antar arah (Cha hanya 11 positif) |

| Kriteria distilasi (praregistrasi) | Hasil |
|---|---|
| (1) p label disesuaikan < 0,01 | HeAR dan OPERA-CT lolos |
| (2) p Δ < 0,05 dan batas bawah CI Δ > 0 | **Gagal** (CI mencakup 0) |
| (3) leave-one-clinic-out tidak runtuh di kedua arah | **Gagal** |

### 7.6 Distilasi HeAR ke CNN golden (eksploratori)

Dijalankan atas permintaan, di luar kriteria di atas. Murid berarsitektur hardware; target 32 komponen PCA dari embedding HeAR,
tanpa label TB; PCA dan murid dilatih hanya pada peserta lipatan latih; probe LR di dalam lipatan.

| Model | AUROC strat. | Nol p95 / maks | Δ di atas klinis |
|---|---|---|---|
| Guru HeAR (klip yang sama) | 0,672 | 0,606 / 0,675 | +0,024 |
| **Murid distilasi (16-d)** | **0,654** | 0,580 / 0,642 | +0,015 |
| Kontrol: jaringan acak tanpa pelatihan | 0,598 | 0,583 / 0,627 | +0,000 |
| Kontrol: target teracak | 0,529 | 0,568 / 0,618 | -0,011 |
| `model_v2` (supervised dari nol) | ~0,56 | - | - |

Murid memprediksi jenis kelamin 0,87 (guru 0,95) dan klinik Kan 0,66 (guru 0,90); AUROC dalam Kan 0,735. Fidelitas komponen PCA
rendah (R² 0,16). Kuantisasi int8 turun 0,008 (lulus); RTL `cnn_top` dan `suaratb_top` bit-exact dengan bobot ini.

## 8. Keputusan penting

| Kode | Keputusan |
|---|---|
| K-04 | Lebar FFT 24 bit, skala 1/2 di semua stage |
| K-14 | Offset fitur dari audio nyata (6690 -> 8265) |
| K-15 | TBscreen sebagai data tambahan, bukan CODA (akses) |
| K-17 | CNN output-stationary, paralel di kanal keluaran |
| K-21 | Target akurasi 0,70/0,80 diganti pelaporan terukur + pembanding WHO TPP |
| K-22 | Clock inti 147,456 MHz (BCLK = clk/144, UART 1280 clk/bit) |
| K-23 | Embedding pralatih diuji; kriteria distilasi tidak terpenuhi; distilasi dicoba eksploratori |

## 9. Keterbatasan

1. **Data kecil dan satu populasi:** 60 positif audio, 3 klinik di Lusaka, tanpa data independen; CI lebar (±0,08); selisih di
   bawah ±0,05 tidak dapat dibuktikan.
2. **Perancu:** jenis kelamin dan klinik; embedding memprediksi keduanya. Strata dikendalikan tetapi tidak menghilangkan risiko.
3. **Mikrofon:** semua data dari perekam dan telepon; belum ada data INMP441; `FEAT_OFFSET` dan detektor provisional.
4. **Detektor tanpa ground truth batuk:** presisi tidak terukur; klip bisa memuat bicara/derau.
5. **Hardware:** hanya sintesis OOC dan simulasi; tidak ada pin, MMCM, atau uji di papan; daya vectorless.
6. **Pemilihan model pada data yang sama:** semua percobaan di §7.4-7.6 memakai 364 peserta yang sama; konfirmasi hanya sah
   pada data baru.
7. **Standar WHO:** tidak dipenuhi; audio tidak menutup jarak itu.

## 10. Peta jalan

| Langkah | Tujuan |
|---|---|
| Data besar (CODA 300+ positif) atau pengumpulan prospektif dengan INMP441 | Satu-satunya jalan mengubah kesimpulan "belum terbukti" |
| Validasi HeAR/distilasi pada klinik/populasi independen | Menjawab jalan pintas klinik dan nilai tambah di atas gejala klinis |
| Seed sweep, constraint I/O, papan panitia (Fase 7); regresi terpadu Fase T sudah lulus di simulasi | Menutup batas hardware |
| Benchmark vs CPU (Fase 8) | Angka akhir latensi/energi |

## Lampiran: lokasi bukti

| Topik | Berkas |
|---|---|
| Spesifikasi, parameter | `golden/params.py`, `docs/vault/spesifikasi.md` |
| Keputusan | `docs/vault/keputusan.md` |
| Verifikasi | `tb/`, `docs/vault/verifikasi.md` |
| Sintesis | `target/`, `docs/vault/sintesis.md` |
| Data | `docs/vault/data.md`, `model/data/` |
| Model dan evaluasi | `docs/vault/model.md`, `model/` |
