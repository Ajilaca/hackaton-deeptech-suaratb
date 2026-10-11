# Model SuaraTB: arsitektur, capaian, batasan

Satu-satunya dokumen tentang model. Angka berasal dari skrip di `model/` (dijalankan 2026-10-11); sumber tiap angka disebut.
Bobot, fitur, dan checkpoint ada di `/data/suaratb` dan tidak masuk repo. Keputusan terkait: K-14 s.d. K-17 di `keputusan.md`.

## Ringkasan (apa adanya)
- **Hardware/pipeline: valid.** Golden int8 bit-exact; kuantisasi tidak menurunkan AUROC (+0,001).
- **Akurasi klinis dari audio: TIDAK didukung data.** Batuk saja 0,56 (CNN) dan 0,66-0,70 (LR statistik log-mel, sebagian besar
  jenis kelamin; residu ≈ 0,58-0,60). Target 0,70 batuk-saja dan 0,80 fusi **tidak tercapai dan tidak diklaim**.
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

Target di CLAUDE.md: batuk-saja ≥ 0,70 **belum**; fusi ≥ 0,80 **belum** (terbaik 0,768); penurunan kuantisasi ≤ 0,02 **lulus**;
HW vs golden bit-exact **lulus di golden** (RTL belum disimulasikan; Fase T).

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

## 6. Batasan (wajib dibawa ke proposal)
1. **Tidak ada klaim akurasi klinis.** AUROC batuk-saja tidak dapat dibedakan dari nol permutasi; setelah jenis kelamin dan
   usia dikendalikan sisa ≈ 0,58-0,60 (di batas keberartian untuk 58 positif).
2. **Jumlah positif kecil** (58): CI ±0,07-0,08. Selisih kecil tidak dapat dipercaya.
3. **Perancu**: jenis kelamin (Zambia); cohort rekrutmen (TBscreen). Uji permutasi dalam-klinik/cohort tidak mengendalikannya.
4. **Tidak ada ground truth batuk**: presisi detektor energi tidak terukur; klip bisa memuat bicara/derau.
5. **Perangkat berbeda**: perekam 44,1 kHz dan telepon 192 kHz; belum ada data dari INMP441. `FEAT_OFFSET` provisional (K-14).
6. **Satu populasi dan sedikit klinik** (3 klinik Lusaka); tidak ada uji prospektif; lintas-dataset tidak bermakna.
7. **Fusi dengan gejala klinis** adalah perhitungan di host, bukan di perangkat; gejala saja sudah 0,73.
8. Hanya VGGish yang diuji sebagai model pralatih (bukan CLAP/HeAR/OPERA); tidak dapat disimpulkan bahwa model lain gagal.
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
- Akses: CODA butuh persetujuan/surat institusi sehingga tidak tersedia untuk iterasi ini (K-15).
Jadi dataset tambahan adalah jalur penguatan yang masuk akal pasca-hackathon, bukan alasan untuk mengklaim akurasi sekarang.

