# Model SuaraTB: arsitektur, capaian, batasan

Satu-satunya dokumen tentang model. Angka berasal dari skrip di `model/` (dijalankan 2026-10-11); sumber tiap angka disebut.
Bobot, fitur, dan checkpoint ada di `/data/suaratb` dan tidak masuk repo. Keputusan terkait: K-14 s.d. K-17 di `keputusan.md`.

## Ringkasan (apa adanya)
- **Hardware/pipeline: valid.** Golden int8 bit-exact; kuantisasi tidak menurunkan AUROC (+0,001).
- **Akurasi klinis dari audio: TIDAK didukung data.** Batuk saja 0,56 (CNN) dan 0,66-0,70 (LR statistik log-mel, sebagian besar
  jenis kelamin; residu ≈ 0,58-0,60). Target 0,70 batuk-saja dan 0,80 fusi **tidak tercapai dan tidak diklaim**.
- **Gejala klinis saja AUROC 0,77 (1.828 peserta, 205 positif); audio tidak menambah apa pun** (Δ +0,003, CI -0,02..+0,03).
  Standar WHO TPP triase (sens ≥ 90%, spes ≥ 70%) **tidak dipenuhi** dan tidak diklaim (lihat §5c).
- **Embedding pralatih (HeAR, OPERA) memberi sinyal audio di atas nol permutasi** setelah jenis kelamin dikendalikan (HeAR AUROC strat.
  0,71, p disesuaikan 0,002), tetapi tambahan di atas klinis kecil dan tidak presisi (Δ +0,04, CI mencakup 0), dan tidak konsisten
  lintas klinik (§5d). Distilasi ke CNN int8 **dicoba** sebagai percobaan eksploratori: murid mewarisi sebagian besar
  sinyal guru (AUROC strat. 0,654 vs guru 0,672), batas klaimnya sama (§5e, K-23).
- Hasil ini sejalan dengan literatur (lintas-dataset sering < 0,6; Zhang dkk., arXiv 2608.25846). Alat adalah triase
  pendukung, bukan diagnosis.

## 1. Model

| Bagian | Isi |
|---|---|
| Masukan | segmen 50 frame x 40 mel, int8: `q = sat8(round((L - 8265) / 2^5))`, L = log2 Q6.8 dari FFT 512 (W=24) |
| Jaringan | Conv3x3 16 -> Conv3x3 32 -> Conv3x3 32 (tiap lapis requant, ReLU, MaxPool 2x2), GAP, FC 32->16 (ReLU), FC 16->1 |
| Ukuran | 14.496 bobot int8, 97 bias, 3.698.448 MAC per klip |
| Aritmetika | akumulator 24 bit bertanda; requant `(acc*M0 + 2^(s-1)) >> s`, M0 Q15 per layer; keluaran logit int16 |
| Skor peserta | rata-rata logit 3-5 batuk (kantong 4 klip saat pelatihan) |
| Spesifikasi bit-exact | `golden/blocks/cnn.py` (QModel/QLayer); bobot terlatih: `/data/suaratb/runs/model_v2` (`.mem`, `qmodel.npz`) |

Keluaran per layer (model_v2):

| layer | bobot | M0 (Q15) | shift | puncak abs(acc) teramati | batas 24 bit |
|---|---|---|---|---|---|
| conv1 | 144 | 28771 | 23 | 58.987 | 8.388.608 |
| conv2 | 4.608 | 21979 | 24 | 177.492 | 8.388.608 |
| conv3 | 9.216 | 18913 | 25 | 257.464 | 8.388.608 |
| fc1 | 512 | 22111 | 20 | 32.871 | 8.388.608 |
| fc2 | 16 | 32307 | 26 | 17.774 | 8.388.608 |

## 2. Data

| Sumber | Peserta | Positif TB | Catatan |
|---|---|---|---|
| Zambia CIDRZ (Kaggle `googlehealthai/google-health-ai`, ODbL-1.0; syarat tambahan belum dibaca) | 360 berlabel (dari ±380 dengan audio) | 58 | Klinik Kan 215/47, Cha 111/10, Chai 34/1 (peserta/positif). Perekam dan Phone A dipakai untuk pelatihan; 7.187 klip batuk dari detektor golden. Label `ground_truth_tb` (Xpert/kultur). Phone B/C: peserta yang sama (hanya variasi perangkat) |
| TBscreen (Nairobi, UW; CC BY 4.0; Sharma dkk. 2024) | 123 dengan audio | 87 | Hanya WAV + CSV (1,37 GB dari zip 424 GB). Cohort A (rekrutmen TB) 93 / cohort C (non-TB) 30: **cohort ≈ label** |
| CODA TB DREAM | - | - | Tidak dipakai: butuh surat institusi dan persetujuan (K-15) |

Pemisahan selalu **per peserta**, bukan per klip. Resample ke 16 kHz; segmen 8.240 sampel dengan onset di frame 5.

## 3. Pelatihan dan kuantisasi

- **MIL**: kantong 4 klip, rata-rata logit (sama seperti perangkat). CV 2x5-lipat per peserta, ensemble seed, early stopping
  pada validasi dalam, augmentasi gain/geser/SpecAugment.
- **Kuantisasi**: PTQ dengan kalibrasi requant; QAT dipakai pada satu lipatan (lipatan 3).
- **Kuantisasi per lipatan** (golden vs float, out-of-fold): korelasi logit 0,9907 / 0,9967 / 0,9991 / 0,9969 / 0,9977; RMSE
  logit 0,001-0,019.
- **AUROC peserta (satu seed)**: float 0,545, int8 golden 0,546: penurunan +0,001 (target ≤ 0,02: lulus). Model akhir: seed 2,
  korelasi logit float vs int8 0,9913.
- Catatan: logit model akhir hampir konstan (-1..8 pada Q8); angka kuantisasi lulus tetapi tidak membuktikan kemampuan klinis.

## 4. Capaian (hasil dalam-dataset Zambia, 360 peserta / 58 positif)

CI 95% bootstrap atas peserta. "Nol" = label diacak dalam klinik dan seluruh pipeline diulang.

| Model | AUROC | CI 95% | Catatan |
|---|---|---|---|
| CNN MIL awal, batuk saja | 0,562 | 0,48-0,64 | nol 0,523 (p95 0,591); p-empiris 0,41; Phone A 0,493, perekam 0,568 |
| CNN MIL diperbaiki (kurangi derau, klip mirip-batuk, epoch tetap) | 0,45-0,49 | - | di bawah acak |
| LR statistik log-mel (80 fitur), semua klip | 0,664-0,673 | 0,61-0,76 | nol 0,54 (p95 0,60-0,63); sebagian besar jenis kelamin |
| LR statistik log-mel, frame aktif saja | 0,611 | 0,54-0,69 | residu sex+usia 0,577 |
| Audio saja, residu sex+usia (LR) | 0,598 | 0,53-0,70 | |
| Jenis kelamin + usia saja | 0,654 | 0,58-0,72 | pembanding perancu |
| Gejala klinis 14 fitur (LR) | 0,730 | 0,67-0,82 | pembanding |
| Klinis diperluas (+berat, tinggi, jumlah gejala, interaksi) | 0,743 | 0,68-0,83 | Δ +0,012 (CI -0,017..+0,042) |
| Klinis diperluas, GBM dangkal | 0,730 | 0,66-0,81 | |
| **Fusi klinis diperluas + audio** | **0,768** | 0,70-0,84 | Δ vs klinis14 +0,038 (CI -0,005..+0,061); skor audio dihitung di dalam lipatan |

Target awal proyek: batuk-saja ≥ 0,70 **belum**; fusi ≥ 0,80 **belum** (terbaik 0,768 pada 360 peserta; 0,78 bila klinis
dilatih pada peserta tanpa audio, §5b). Angka 0,70/0,80 adalah tebakan awal kita (spesifikasi awal, "nilai awal"), bukan syarat lomba
atau standar medis; pembanding klinis yang benar adalah WHO TPP (§5c); penurunan kuantisasi ≤ 0,02 **lulus**;
HW vs golden bit-exact **lulus** (RTL disimulasikan per modul dan ujung ke ujung dataset + mikrofon, 2026-10-11; regresi penuh Fase T belum).

Perancu jenis kelamin (Zambia):

| Pemeriksaan | Hasil |
|---|---|
| Prevalensi TB laki-laki vs perempuan | 25% (n=175) vs 8% (n=185) |
| Jenis kelamin saja -> TB | AUROC 0,652 |
| Audio -> jenis kelamin | AUROC 0,891 (log-mel), 0,813 (VGGish) |
| Audio saja dalam laki-laki / perempuan | 0,543 / 0,424 |
| Audio setelah sex+usia dikendalikan | ≈ 0,58-0,60 |

## 5. Uji lintas-dataset dan model pralatih

| Uji | n / TB | AUROC | CI 95% | Arti |
|---|---|---|---|---|
| TBscreen, cohort saja (perancu) | 123 / 87 | 0,917 | 0,85-0,97 | label ≈ cohort |
| TBscreen, audio dalam-dataset | 123 / 87 | 0,577 | 0,47-0,69 | nol 0,532 (p95 0,627): tidak ada sinyal |
| TBscreen, audio, laki-laki / perempuan | 86 / 65, 37 / 22 | 0,75 / 0,76 | 0,61-0,88 / 0,58-0,92 | cohort tetap perancu; 7 irisan dilihat |
| Zambia -> TBscreen (z-norm) | 123 / 87 | 0,53-0,55 | mencakup 0,5 | tidak berpindah |
| TBscreen -> Zambia (z-norm) | 360 / 58 | 0,50-0,52 | mencakup 0,5 | tidak berpindah |
| VGGish pralatih, Zambia dalam-dataset | 360 / 58 | 0,510 (residu 0,451) | 0,43-0,60 | lebih buruk dari log-mel |
| VGGish, Zambia -> TBscreen | 123 / 87 | 0,669 (residu 0,624) | 0,56-0,77 | kemungkinan perbedaan cohort |
| VGGish, TBscreen -> Zambia | 360 / 58 | 0,548 | 0,46-0,63 | bebas perancu cohort: tidak beda dari acak |

Distilasi VGGish -> CNN int8 **dibatalkan** (K-16): guru tidak memberi sinyal TB yang andal setelah sex/usia dikendalikan.

### 5a. Lintas-perangkat (peserta Zambia yang sama: perekam, Phone A, B, C; 2026-10-11)

LR statistik log-mel (frame aktif), split 5-lipat per peserta (peserta tidak muncul di latih dan uji sekaligus), 3 ulangan.
Jumlah peserta per perangkat: perekam 244 (34 positif), Phone A 341 (54), B 347 (58), C 355 (60). Skor ulangan ditumpuk,
jadi CI sempit; baca sebagai indikasi.

| Uji | AUROC | Catatan |
|---|---|---|
| Dalam-perangkat: perekam / A / B / C | 0,539 / 0,612 / 0,624 / 0,616 | CI 95% mencakup 0,5 hanya untuk perekam; selebihnya ≈ 0,53-0,69 |
| Latih satu perangkat -> uji perangkat lain (12 pasangan) | 0,520-0,595 (median ≈ 0,58) | tidak lebih baik dari dalam-perangkat |
| Latih semua perangkat -> uji per perangkat | 0,583 / 0,631 / 0,607 / 0,592 | menambah perangkat tidak mengubah apa pun secara berarti |

Kesimpulan: sinyal audio lemah (≈ 0,58-0,63) dan tidak naik melewati 0,70 dengan Phone B/C. Pada uji ini jenis kelamin/usia
tidak dipisahkan; §4 menunjukkan sebagian besar sinyal LR berasal dari keduanya. Tidak ada klaim ketahanan lintas-perangkat.
Kode: `model/evaluation/device_eval.py`; berkas laporan lengkap ada di `$SUARATB_DATA/runs/reports/lintas-perangkat.md`.

### 5b. Klinis pada semua peserta berlabel dan manfaat tambahan audio (2026-10-11)

Metadata Zambia memuat 1.828 peserta berlabel (205 positif); hanya 360 (58 positif) punya audio yang dipakai. Fitur klinis
(gejala, umur, berat, dst.; tanpa xpert/kultur/smear/AI X-ray) tidak butuh audio, jadi dapat dievaluasi pada semuanya.

| Uji | n / positif | AUROC | CI 95% |
|---|---|---|---|
| Klinis 14 fitur, semua peserta (CV 5x5) | 1.828 / 205 | 0,772 | 0,74-0,81 |
| Klinis diperluas, semua peserta | 1.828 / 205 | 0,772 | 0,74-0,81 |
| Klinis diuji pada 360 berlaudio (dilatih pada 1.468 tanpa audio) | 360 / 58 | 0,774-0,782 | 0,70-0,85 |
| Audio saja (log-mel, LR) pada 360 | 360 / 58 | 0,684 | 0,61-0,76 |
| **Klinis + audio (fusi) pada 360** | 360 / 58 | 0,776 | 0,70-0,84 |
| Δ fusi vs klinis | | **+0,003** | -0,021..+0,026 |

- Klinis saja pada data penuh 0,77, sedikit di atas 0,73 yang dihitung pada 360 peserta (estimasi sebelumnya kurang tepat).
- **Audio tidak menambah apa pun di atas klinis**: Δ +0,003, CI mencakup 0. Audio saja 0,68 sebagian besar dari
  jenis kelamin/usia (§4).
- Peserta berlaudio tidak mewakili semuanya: prevalensi 16,1% vs 10,0% (klinik Kan 21,9% vs 13,4%).
- Target fusi ≥ 0,80 tidak tercapai (terbaik 0,78). Skor klinis diperoleh dari model yang dilatih pada peserta tanpa audio;
  skor audio dihitung di dalam lipatan (tanpa kebocoran). Kode: `model/clinical_models/clinical_full.py`.

### 5c. Jarak ke standar WHO TPP (triase berbasis komunitas)

Standar: sensitivitas ≥ 90% dan spesifisitas ≥ 70% (WHO 2014; dikutip Huddart dkk., Sci. Data 2024). Skrining gejala dengan
kuesioner pada pedoman WHO 2021: sensitivitas 42%. Titik operasi dari skor klinis (CV 5x5, 1.828 peserta; bukan dari audio):

| Syarat | Hasil kita |
|---|---|
| Sensitivitas ≥ 90% | spesifisitas **38%** (butuh 70%) |
| Sensitivitas ≥ 80% | spesifisitas 62% |
| Spesifisitas ≥ 70% | sensitivitas **71%** (butuh 90%) |

- Untuk 90%/70% sekaligus, AUROC harus ≈ **0,90** (aproksimasi binormal; perkiraan, bukan angka dari paper). Kita 0,77.
- Audio tidak menutup jarak itu (§5b), dan fitur klinis tambahan di Zambia tidak menambah (klinis diperluas 0,772 = klinis 14).
- Fitur klinis yang dipakai (14): umur, jenis kelamin, BMI, lama batuk, batuk berdahak, hemoptisis, nyeri dada, sesak, demam,
  keringat malam, penurunan berat badan, status HIV, riwayat TB, merokok. Suhu dan denyut nadi **tidak** ada di Zambia
  (variabel CODA). Alat hanya punya mikrofon: semua fitur klinis harus diinput manual, bukan diukur alat.
- Kolom yang tidak boleh dipakai: xpert, kultur, smear, LAM, prediksi AI X-ray, ground truth (kebocoran label).
- Populasi Zambia = pasien bergejala; TPP komunitas merujuk populasi lain, jadi perbandingan ini hanya orientasi.

Posisi yang dapat dipertahankan: alat **prioritas rujukan**, bukan pengganti tes dan tidak memenuhi TPP. Memenuhi TPP butuh
data yang lebih besar dan beragam serta bukti sinyal akustik yang belum ada pada data ini.

### 5d. Embedding pralatih dengan protokol evaluasi tetap (2026-10-12; K-23)

Audit data: rilis Kaggle memuat 1.481 berkas audio (sama dengan yang diunduh); angka "664 peserta" pada deskripsi dataset tidak ada
di rilis. Peserta berlabel dengan audio: **364 (60 positif)**; pemulihan barcode salah ketik tidak menambah peserta.

Protokol: unit = peserta; CV 5-lipat, stratifikasi label x klinik x jenis kelamin; skor klinis dilatih hanya pada peserta tanpa audio;
metrik utama = AUROC terstratifikasi jenis kelamin dan Δ fusi - klinis; nol = label diacak dalam klinik x jenis kelamin dengan seluruh
pipeline diulang. Embedding: HeAR (jendela 2 s di sekitar onset detektor golden, maks 6 klip per peserta), OPERA-CE dan OPERA-CT
(rekaman utuh). Klasifikasi: LR linear. Semua kandidat dilaporkan.

| Fitur | AUROC strat. jenis kelamin | Nol p95 / maks | Δ vs klinis (CI 95% bootstrap) |
|---|---|---|---|
| Statistik log-mel (pembanding) | 0,599 | 0,619 / 0,672 | -0,002 (-0,032..+0,029) |
| HeAR | 0,675 | 0,561 / 0,582 | +0,015 (-0,023..+0,058) |
| OPERA-CT | 0,677 | 0,609 / 0,644 | +0,029 (-0,011..+0,068) |
| OPERA-CE | 0,606 | 0,604 / 0,612 | +0,013 (-0,013..+0,041) |

Uji permutasi final (1.000 permutasi, statistik maksimum atas tiga kandidat, permutasi sama untuk semua kandidat, C tetap 0,03
ditetapkan sebelum angka final; hasil serupa untuk C = 0,01 dan 0,1):

| Kandidat | AUROC strat. | p disesuaikan (label) | Δ di atas klinis | p disesuaikan (Δ) | CI bootstrap Δ |
|---|---|---|---|---|---|
| HeAR | 0,709 | 0,002 | +0,039 | 0,003 | -0,011..+0,092 |
| OPERA-CT | 0,664 | 0,006 | +0,025 | 0,015 | -0,017..+0,064 |
| OPERA-CE | 0,621 | 0,089 | +0,022 | 0,023 | -0,003..+0,048 |

Pemeriksaan:
- **Kontrol negatif** (fitur Gaussian acak, dua seed): tidak ada p disesuaikan mendekati 0,05; uji tidak menghasilkan positif palsu.
- **Lintas-perangkat** (HeAR, peserta sama, Phone A/B/C): AUROC strat. 0,62-0,69 di 9 pasangan latih/uji. Menunjukkan ketahanan terhadap
  mikrofon, bukan terhadap populasi (peserta dan isi batuk sama).
- **Jalan pintas**: embedding HeAR memprediksi jenis kelamin AUROC 0,955 dan klinik (Kan vs lain) 0,875. AUROC skor audio dalam strata:
  Kan 0,728 (49 positif), Cha 0,644 (11), laki-laki 0,691, perempuan 0,632 (15).
- **Leave-one-clinic-out** (strat. jenis kelamin): Kan->Cha HeAR 0,547 / OPERA-CT 0,731 / OPERA-CE 0,638; Cha->Kan HeAR 0,617 /
  OPERA-CT 0,555 / OPERA-CE 0,530. Tidak ada kandidat yang melewati p95 nol di kedua arah; Cha hanya 11 positif (CI +-0,2).

Kesimpulan: ada indikasi sinyal audio dan tambahan kecil di atas gejala klinis, **belum terbukti presisi dan belum konsisten lintas
klinik**; tidak ada data independen. Bukan klaim klinis. Kriteria distilasi yang ditetapkan sebelumnya tidak terpenuhi (K-23);
distilasi tetap dicoba secara eksploratori (§5e).

### 5e. Distilasi HeAR -> CNN golden (eksploratori, 2026-10-12; K-23)

Tujuan: mengetahui seberapa besar sinyal guru yang bisa dipindahkan ke jaringan yang muat di akselerator. Bukan bagian kriteria
praregistrasi. Murid = arsitektur hardware (Conv16/32/32, GAP, FC32->16; 14.496 bobot); masukan log-mel 0,5 s (fitur hardware) vs
guru 2 s. Distilasi memakai target 32 komponen PCA dari embedding HeAR per klip, TANPA label TB; PCA dan murid dilatih hanya pada klip
peserta lipatan latih; probe LR (embedding 16-d rata-rata per peserta) di dalam lipatan protokol E0. 6.354 klip, 364 peserta, 60 positif.

| Model | AUROC strat. jenis kelamin | Nol p95 / maks | p-empiris | Δ di atas klinis |
|---|---|---|---|---|
| Guru HeAR (klip yang sama) | 0,672 | 0,606 / 0,675 | 0,010 | +0,024 |
| **Murid distilasi (16-d)** | **0,654** | 0,580 / 0,642 | 0,005 | +0,015 |
| Kontrol: jaringan acak, tanpa pelatihan | 0,598 | 0,583 / 0,627 | 0,030 | +0,000 |
| Kontrol: target teracak | 0,529 | 0,568 / 0,618 | 0,244 | -0,011 |
| `model_v2` (supervised dari nol, §4) | ≈ 0,56 (mentah) | - | - | - |

Jalan pintas (ulangan 0): embedding memprediksi jenis kelamin 0,867 (guru 0,953; jaringan acak 0,640) dan klinik Kan 0,655 (guru 0,896;
acak 0,586). AUROC skor murid dalam Kan 0,735 (guru 0,725), dalam Cha 0,590 (11 positif), laki-laki 0,638, perempuan 0,687 (15 positif).
Fidelitas komponen PCA pada klip uji rendah (R2 0,16): murid hanya mewarisi sebagian arah variasi guru.

Pembacaan: distilasi mewarisi sebagian besar sinyal guru pada jaringan yang muat di hardware, dan jauh di atas model yang dilatih
dari nol dengan label (0,56); kontrol menunjukkan peningkatan itu berasal dari distilasi, bukan arsitektur. Batasnya sama dengan guru:
tambahan di atas klinis tidak signifikan, 364 peserta dari satu populasi, jenis kelamin tertangkap kuat. Bukan klaim klinis.

Kuantisasi dan hardware (murid + probe terlipat menjadi FC 16->1; PTQ tanpa label; out-of-fold, ulangan 0 protokol E0):

| Ukuran | Float | int8 golden | Selisih |
|---|---|---|---|
| AUROC strat. jenis kelamin | 0,653 | 0,645 | 0,008 (kriteria <= 0,02: lulus) |
| AUROC mentah | 0,729 | 0,723 | 0,006 |

Korelasi logit float vs int8 per lipatan 0,955-0,987 (lebih rendah dari `model_v2`, 0,991-0,999). Model akhir (murid pada semua klip,
probe pada semua peserta): korelasi 0,988; puncak akumulator 138.084 (batas 24 bit 8.388.608). RTL `cnn_top` dengan bobot ini
**bit-exact terhadap golden pada 8 klip nyata** (Icarus, P=16, 2026-10-12). Bobot di `$SUARATB_DATA/runs/model_distill` (tidak di-commit).
`suaratb_top` dengan bobot ini juga **bit-exact terhadap golden** pada mode dataset (UART, 2 kasus) dan mode mikrofon
(I2S -> detektor -> DSP -> CNN -> UART), Icarus, 2026-10-12. Belum dilakukan: sintesis ulang dengan bobot ini (ROM berukuran sama; sumber daya diperkirakan identik).

## 6. Batasan (wajib dibawa ke proposal)
1. **Tidak ada klaim akurasi klinis.** AUROC batuk-saja tidak dapat dibedakan dari nol permutasi; setelah jenis kelamin dan
   usia dikendalikan sisa ≈ 0,58-0,60 (di batas keberartian untuk 58 positif).
2. **Jumlah positif kecil** (58): CI ±0,07-0,08. Selisih kecil tidak dapat dipercaya.
3. **Perancu**: jenis kelamin (Zambia); cohort rekrutmen (TBscreen). Uji permutasi dalam-klinik/cohort tidak mengendalikannya.
4. **Tidak ada ground truth batuk**: presisi detektor energi tidak terukur; klip bisa memuat bicara/derau.
5. **Perangkat berbeda**: perekam 44,1 kHz dan telepon 192 kHz; belum ada data dari INMP441. `FEAT_OFFSET` provisional (K-14).
6. **Satu populasi dan sedikit klinik** (3 klinik Lusaka); tidak ada uji prospektif; lintas-dataset tidak bermakna.
7. **Fusi dengan gejala klinis** adalah perhitungan di host, bukan di perangkat; gejala saja sudah 0,73.
8. Model pralatih yang diuji: VGGish (jendela salah, uji tidak sah), HeAR, OPERA-CE/CT (§5d); CLAP dan lainnya belum. Sinyal HeAR/OPERA
   bercampur dengan klinik dan jenis kelamin (embedding memprediksi keduanya) dan hanya diuji pada 364 peserta dari satu populasi.
9. Lisensi: Zambia ODbL-1.0 (syarat tambahan belum dibaca); TBscreen CC BY 4.0 (wajib sitasi). Bobot belum dipublikasikan.

## 7. Arti bagi proyek
- Akselerator dan golden tetap sah sebagai pipeline lengkap audio -> skor dengan latensi/sumber daya terukur (Fase 6).
- Bobot dapat diganti tanpa mengubah RTL (deskriptor/ROM dari `.mem`); bila data yang lebih baik tersedia, model dilatih ulang.
- Klaim yang boleh: "pipeline lengkap di akselerator, bit-exact terhadap golden, latensi/sumber daya dari laporan alat".
  Klaim yang tidak boleh: akurasi diagnosis, generalisasi lintas-populasi, "pertama di dunia".

## 8. Peluang penguatan model dengan dataset tambahan (belum dilakukan)
Dataset TB-batuk yang lebih besar dan beragam, misalnya **CODA TB DREAM** (±1.000-2.100 peserta, 7 negara, banyak
perangkat), dapat memperkuat pemodelan. Yang diharapkan:
- **Daya statistik**: positif naik dari 58 (Zambia) / 87 (TBscreen, perancu cohort) ke ratusan; CI menyempit dari ±0,07.
- **Kontrol perancu**: variasi perangkat, negara, dan jenis kelamin memungkinkan pemisahan sinyal TB dari artefak akuisisi
  (analisis per perangkat/negara seperti di Zhang dkk., arXiv 2608.25846).
- **Pelatihan lintas-perangkat**: makalah yang sama melaporkan pelatihan multi-perangkat meningkatkan ketahanan pada
  perangkat yang tidak terlihat.
- **Evaluasi eksternal yang bermakna**: ≥ 3 sumber memungkinkan latih-di-dua, uji-di-satu.
- **Pembaruan model tanpa mengubah hardware**: bobot dan tabel requant hanya isi ROM (K-17).

Yang TIDAK dijamin :
- Kenaikan AUROC. Pada CODA sendiri makalah EPFL melaporkan dalam-dataset 0,63-0,64 dan lintas-dataset sering < 0,6;
  prevalensi per negara dapat diprediksi dari akuisisi (shortcut).
- Menghilangnya perancu: label, perangkat, dan negara saling terkait di CODA; butuh balancing per subgrup.
- Akses: CODA butuh akun Synapse Certified + Validated, Intended Data Use Statement, dan persetujuan Terms of Use; upaya
  verifikasi terlalu rumit sehingga tidak tersedia untuk iterasi ini (K-15). Set latih solicited hanya 0,43 GB
  (1.105 peserta, 297 positif), jadi layak dicoba pasca-hackathon.
Jadi dataset tambahan adalah jalur penguatan yang masuk akal pasca-hackathon, bukan alasan untuk mengklaim akurasi sekarang.

