# Keputusan Fase 0

Format: ID, keputusan, alasan, alternatif/risiko. Semua angka berasal dari `python -m golden.calibrate`
(hasil mentah: `golden/out/phase0_calibration.json`, di .gitignore) dan `pytest golden`.
**Seluruh kalibrasi memakai audio SINTETIS** (`golden/synth.py`); dataset Kaggle belum ada di `data/raw`.

## K-01 Aritmetika
Pembulatan = tambah setengah lalu geser aritmetik (floor(v + 0,5)); saturasi eksplisit; int64 di numpy.
Alasan: cocok dengan penambah + pemotongan bit di RTL, tak memerlukan "round-to-even".

## K-02 Butterfly FFT
Satu pembulatan per butterfly: `(a·2^15 ± b·w + 2^(14+s)) >> (15+s)`. Twiddle k=0 = 32767 (1,0 tak terwakili di Q1.15).
Alasan: satu penambah lebar, galat lebih kecil daripada pembulatan ganda. Plateau SNR ~85 dB pada sinyal keras
dibatasi twiddle Q1.15, bukan lebar data (tabel di `kalibrasi-fase0.md`).

## K-03 Segmen 50 frame
CLAUDE.md menyebut "segmen 40x50" dan "45 frame setelah onset". Ditafsirkan 5 frame pre-roll + 45 pasca-onset = 50.
50 frame membutuhkan 8240 sampel (0,515 s); klip dataset 8000 sampel diberi 240 nol di ekor.
Risiko: tafsiran ini asumsi; jika bukan itu maksudnya, ubah `PRE_ROLL`/`POST_ONSET`/`N_FRAMES` (CNN mengikuti).

## K-04 Lebar FFT = 24 bit, skala 1/2 di semua stage  *(mengubah nilai awal 18)*
Aturan yang ditetapkan: lebar terkecil dengan galat fitur log-mel ≤ 0,5 langkah int8 (0,0625 oktaf) pada
semua tingkat derau latar uji (σ = 1,5 sampai 100 LSB), dihitung pada elemen ≥ lantai derau kuantisasi
masukan + 10 dB, dengan jadwal skala yang dijamin tanpa overflow. Hasil (RMS galat, oktaf):

| W | σ=1,5 | σ=3 | σ=10 | σ=30 | σ=100 | SNR FFT σ=3 |
|---|---|---|---|---|---|---|
| 18 | 3,42 | 2,94 | 2,20 | 1,08 | 0,27 | -1,5 dB |
| 22 | 0,125 | 0,116 | 0,085 | 0,039 | 0,013 | 21,7 dB |
| 23 | 0,065 | 0,058 | 0,044 | 0,020 | 0,006 | 27,7 dB |
| **24** | **0,033** | **0,030** | 0,023 | 0,010 | 0,004 | **33,8 dB** |

18 bit gagal telak pada ruang tenang. 23 bit nyaris lolos (gagal di σ=1,5) dan lebar ganjil tak menguntungkan.
24 bit tetap muat satu blok DSP (data 24 bit pada port 25 bit, twiddle 16 bit pada port 18 bit).
Prinsip: derau pembulatan FFT (~0,58 LSB absolut) harus jauh di bawah derau kuantisasi int16 masukan;
pada G=8 selisihnya ~10 dB (menambah ~0,4 dB), pada G=2 (W=18) derau FFT justru lebih besar.
Alternatif yang dicatat: jadwal tanpa skala di 1-3 stage pertama menambah ~6 dB SNR per stage
(W=24: 34,3 -> 40,3 -> 46,3 -> 52,5 dB pada σ=3) tetapi kehilangan jaminan overflow: `first3_0` jenuh
(12 sampel) pada batuk ~-10 dBFS, `first1_0` tidak jenuh pada uji tetapi jaminannya hanya untuk puncak < -6 dBFS.
Dipilih jadwal aman. **Terbuka:** apakah pengguna mau `first1_0` (+6 dB) dengan jaminan lebih lemah.
Catatan: AUROC belum dapat diukur (tidak ada data berlabel), sehingga syarat "AUROC turun ≤ 0,01" BELUM diperiksa.

## K-05 Log2: masukan 58 bit, keluaran Q6.8 (spesifikasi awal: 32 bit, Q5.8)  *(mengubah spesifikasi)*
Dengan W=24 dan penguatan 2^(2G)=2^16, batas atas akumulator mel (Parseval, sinyal skala penuh) = 58 bit.
Memotongnya ke 32 bit memerlukan MEL_SHIFT yang bergantung data (13 bit pada ensemble sintetis) dan menyaturasi
bila gain mikrofon berubah. Leading-one 58 bit + keluaran 14 bit hanya menambah beberapa gerbang dan
menghilangkan satu konstanta yang bergantung data. MEL_SHIFT = 0 dipertahankan sebagai tuas bila nanti ingin dikecilkan.
Akurasi log2: ≤ 2 LSB Q.8 (0,008 oktaf) diuji `test_log2_accuracy`.

## K-06 Kuantisasi fitur ⚠
q = sat8(round((L - 6690) / 32)): langkah 0,125 oktaf (0,38 dB); rentang ±16 oktaf (±48 dB) di sekitar offset.
Ensemble sintetis: persentil 0,5/99,5 = 3380/9999 (L, Q6.8), 0,01% elemen nol. `FEAT_OFFSET` hanya berlaku
untuk gain mikrofon sintetis; HARUS dihitung ulang pada audio INMP441 nyata.

## K-07 Detektor batuk ⚠
Nilai awal 8 (rasio) dan 64 (EMA = 2^6), ≥ 2 frame dipertahankan. Sapuan K∈{4,6,8,12,16} x S∈{4..7} x min-hop∈{2,3}
pada aliran sintetis, aturan (ditetapkan sebelum melihat hasil): FP ≤ 0,5/menit (derau, klik, lompatan level;
nada keras dilaporkan terpisah), recall maksimum pada SNR 10 dB, lalu terdekat dengan nilai awal.
Hasil: **K=8, S=6, min-hop=2** (K=4/6 juga 100% pada 10 dB dan lebih tinggi pada 6 dB dengan FP sama, tetapi
lebih jauh dari nilai awal dan negatif sintetis tidak mewakili latar nyata; recall 100%
pada 10/15/25 dB, 61% pada 6 dB; FP derau 0, klik 0; lompatan level x4 memicu 1x per lompatan; nada keras 4,7/menit
tak dapat ditolak detektor energi, begitu pula bicara, dan diserahkan ke CNN).
Tafsiran "tahan 300 ms" = refractory 30 frame setelah segmen selesai. N dibekukan saat merekam; lantai N = 640 (derau ~2 LSB, asumsi).
Risiko: seluruh angka bergantung pada generator sintetis; tidak boleh dikutip sebagai kinerja detektor.

## K-08 Pre-emphasis tanpa pembulatan int16  *(menambah spesifikasi)*
Keluaran pre-emphasis 16+8 bit (bit pecahan = guard FFT). Diukur pada W=24, RMS galat fitur vs float
(6 klip batuk sintetis, 20 dB, elemen ≥ lantai derau masukan + 10 dB; oktaf):

| σ derau latar | pembulatan int16 | 16+8 bit |
|---|---|---|
| 1,5 | 0,066 | 0,029 |
| 3 | 0,064 | 0,027 |
| 10 | 0,120 | 0,024 |
| 30 | 0,044 | 0,011 |

Dengan int16 galat melewati ambang 0,0625 oktaf (K-04) pada σ ≥ 3; dengan 16+8 bit di bawahnya semua.
Biaya: window memakai operan 24 bit x 16 bit (1 DSP). Detektor memakai bagian bulat (pembulatan ke int16)
agar skalanya tak bergantung lebar FFT.

## K-09 CNN
Padding "same" nol; MaxPool lantai (25 -> 12, 12 -> 6): peta akhir 5x6, sesuai hitungan 3.698.448 MAC di CLAUDE.md.
ReLU+saturasi [0,127] digabung dengan requant (aktivasi tersembunyi non-negatif). Logit akhir int16 (agar tidak
kehilangan resolusi untuk rata-rata 3-5 batuk). GAP lewat pengali resiprokal Q15. Bias int32 yang harus muat 24 bit.

## K-10 Akumulator 24 bit cukup
Fan-in terbesar 288 (conv3). |acc| ≤ 288·127·128 = 4.681.728 tanpa bias; batas 2^23 = 8.388.608; sisa untuk bias ≈ 3,7 juta.
Diuji analitik dan dengan masukan ekstrem (bobot semua -128). `AccOverflow` dilempar bila bias melanggar.

## K-11 Mel
HTK (2595·log10(1+f/700)), segitiga puncak 1 di domain Hz tiap bin, bobot Q0.12 (puncak dipotong 4095).
Tidak ada band kosong; tiap bin masuk ≤ 2 band; ROM sparse 493 kata x 19 bit {band, w, eob}.

## K-12 Anggaran
`budget.py` menghitung bit logis, blok per memori, pengali per fungsi (pemecahan operan DSP), dan siklus ideal.
Hasil P=16: 282 Kbit, 31 blok pengali, 231.168 siklus. EP4CE6 TIDAK muat konfigurasi dasar (memori 105%, pengali 41 vs 15).
Dengan kanal 8/16/16 memori muat (56%) tetapi pengali masih di atas 15 blok keras; perlu pengali di LUT.
Kapasitas perangkat dari ingatan, belum diverifikasi.

## Perubahan terhadap CLAUDE.md
| Item | Awal | Sekarang |
|---|---|---|
| Lebar FFT | 18 (24 bila SNR kurang) | 24 (K-04) |
| log2 | Q5.8, masukan 32 bit | Q6.8, masukan 58 bit (K-05) |
| Pre-emphasis | tak tentu | 16+8 bit (K-08) |
| Detektor 8 / 64 | tebakan | dipertahankan, berbasis sintetis (K-07) |

## K-13 Fase 2: antarmuka front-end (ditulis, belum dijalankan)
- `i2s_rx`: FPGA sebagai master (BCLK = clk/96, WS = BCLK/64). 24 bit MSB-first, tertunda 1 BCLK; keluaran = 16 bit teratas
  (pemotongan/lantai). Konsekuensi: derau kuantisasi masukan = int16 seperti diasumsikan golden; bit rendah mikrofon (24 bit)
  tidak dipakai (lihat butir terbuka di status-fase.md).
- `preemph`: SATU perkalian a·x[n-1] menghasilkan dua keluaran: `out_wide` (16+8 bit, jalur FFT, K-08) dan `out_s16`
  (int16, jalur detektor). Keduanya sama persis dengan golden `preemph(x, frac=8)` dan `preemph(x)`. Pembulatan ganda
  TIDAK diganti dengan pembulatan `out_wide` ke 16 bit karena hasilnya bisa berbeda 1 LSB.
- `frame_buf`: cincin 512 x 24 bit; frame diputar ulang 400 sampel/siklus-tunggal setelah sampel pemicu; energi hop
  dihitung di jendela [160f+240, 160f+400) = akhir frame f (golden `hop_energy`). Kontrak: jarak sampel >= 404 siklus.
- `cough_det`: antarmuka event (`trig`, `onset_frame`, `start_frame`, `end_frame`, `seg_done`); nilai pembekuan N saat merekam
  dan refractory identik golden. `seg_done` memberi tahu feat_buf (Fase 3) bahwa 50 kolom terakhir = segmen.

## K-14 FEAT_OFFSET dari audio nyata (6690 -> 8265)
Dihitung oleh `golden.calibrate_real` pada 1.764 klip batuk nyata (perekam). Dengan 6690 (sintetis) 2,0% elemen jenuh
int8, dengan 8265 0,2%. Model CNN Fase M dilatih pada fitur dengan offset ini, sehingga `params.py`, tabel, paket SV,
dan vektor ikut diubah. PROVISIONAL: gain mikrofon INMP441 berbeda; harus dihitung ulang pada audio perangkat.
FFT_W tetap 24 (pada data nyata W=22 juga lolos ambang K-04, 0,029 oktaf; belum diputuskan).

## K-15 Sumber data tambahan: TBscreen, bukan CODA (2026-10-11)
- CODA TB DREAM tidak dipakai: akses Synapse butuh akun Certified + Validated, Intended Data Use Statement (≤ 500 kata),
  dan persetujuan Terms of Use (Huddart dkk., Sci. Data 2024). Pengguna sudah mencoba; verifikasinya terlalu rumit untuk
  selesai sebelum Demo Day. Set latih "solicited" hanya 0,43 GB (1.105 peserta, 297 positif), jadi bila akses diperoleh
  kelak unduhannya cepat.
- Kriteria data tambahan: unduhan terbuka tanpa persetujuan, label TB per peserta, audio batuk WAV yang bisa di-resample
  ke 16 kHz, dan lisensi yang jelas.
- Terpilih: **TBscreen** (Sharma dkk., Sci. Adv. 2024; Nairobi; CC BY 4.0). Hanya WAV + CSV yang diambil (1,37 GB) dari
  zip 424,6 GB lewat HTTP range request. Klip 1 detik (batuk pasif dan paksa), tiga perangkat (Pixel, mik boundary, Yeti).
- Batas: populasi, perangkat, dan protokol berbeda dari Zambia. Makalah lintas-dataset (Zhang dkk., arXiv 2608.25846)
  melaporkan AUROC lintas-dataset sering < 0,6, jadi TBscreen dipakai untuk (a) evaluasi dalam-dataset kedua dan
  (b) uji lintas-dataset yang jujur, bukan untuk klaim generalisasi.
- Kandidat lain yang diperiksa: COUGHVID/Coswara (terbuka, tetapi COVID, tanpa label TB); halaman MakBRC Uganda
  sedang offline. Tidak ada dataset TB terbuka lain yang ditemukan.

## K-16 Hasil langkah 1-3: tidak ada sinyal audio-TB yang andal; distilasi dibatalkan (2026-10-11)
- TBscreen (`model.md` bagian 5): dalam-dataset 0,577 (nol p95 0,627); lintas-dataset 0,50-0,55; cohort A/C hampir = label
  (cohort saja 0,917), jadi angka dalam-TBscreen tidak dapat ditafsirkan sebagai penanda TB.
- Klinis diperluas (`model.md` bagian 4): 0,743 vs 0,730 (Δ +0,012, CI mencakup 0); fusi terbaik 0,768; target 0,80 belum tercapai.
- VGGish (`model.md` bagian 5): lebih buruk daripada log-mel di Zambia; distilasi tidak dilakukan.
- Dampak: target batuk-saja 0,70 dan fusi 0,80 tidak dicapai dan tidak diklaim. Klaim proyek tetap: pipeline lengkap dengan
  latensi/sumber daya terukur. Opsi revisi target (laporan CI + p-permutasi setelah kontrol sex/usia) menunggu keputusan pengguna.

## K-17 Arsitektur Fase 4: CNN output-stationary, paralel di kanal keluaran (2026-10-11)
- P lane = P kanal keluaran untuk piksel yang sama; 1 aktivasi/siklus dibroadcast; bobot P x 8 bit per
  siklus dari ROM; akumulator 24 bit; MaxPool pada akumulator sebelum requant (bit-exact dengan golden karena M0 >= 0 dan
  ReLU monoton); satu unit requant dipakai bergiliran; FC memakai pe_array yang sama (k=1); GAP unit kecil; padding lewat
  flag luar-batas; deskriptor layer dari ROM `.mem` sehingga model dapat diganti tanpa mengubah RTL.
- Alasan: 1 baca aktivasi/siklus (tanpa RAM berbank/line buffer), siklus = 231.168 pada P=16 (sesuai budget.py), risiko
  jadwal rendah. Ditolak: systolic/weight-stationary (kontrol rumit), paralel-piksel (butuh P aktivasi/siklus),
  akumulator + LR (bukan akselerator; hanya disebut sebagai alternatif hemat di proposal).
- Akurasi TIDAK diklaim (K-16); arsitektur dipilih demi demo akselerator yang terukur.

## K-18 Fase 3: antarmuka dan jadwal DSP (ditulis, belum disimulasikan)
- Rantai: `frame_buf` -> `window` -> `fft512` -> `power` -> `mel` -> `log_q` -> `quant` -> `feat_buf` (`dsp_chain` membungkus
  window..quant). Semua parameter dari `suaratb_pkg`; ROM dari `rtl/mem/*.mem` (parameter `INIT_*`).
- Laju: waktu per frame tersedia 983 rb siklus (hop 160 sampel @ 98,304 MHz), kebutuhan ~6-8 rb siklus, jadi seluruh rantai
  dibuat SERIAL dan kecil: `fft512` 1 butterfly, 2 siklus/butterfly, RAM in-place 512 x 48 bit; jeda `DRAIN`=8 antar-stage
  (aman; urutan indeks sudah cukup, margin sengaja); keluaran bin tiap `OUT_GAP`=4 siklus (kontrak: `power` >= 1, `mel` >= 3).
- Perkiraan sumber daya (ESTIMASI, bukan laporan alat): pengali window 1, fft 4 (24x16), power 2 (24x24), mel 1 (48x12);
  RAM FFT 24 Kbit, RAM fitur 20 Kbit, akumulator mel 40 x 58 bit di register.
- Verifikasi tanpa simulasi RTL (aturan 3): model Python jadwal FFT (`golden/test_fft_schedule.py`: bit-exact vs golden,
  waktu tulis a'/b', pengalamatan in-place, indeks twiddle, geser stage) dan model alamat window/mel/feat_buf
  (`golden/test_dsp_addressing.py`). Testbench cocotb ditulis (`tb/test_{window,fft512,power,mel,log_q,quant,dsp_chain,
  feat_buf}.py`) dan baru dijalankan di Fase T. RTL belum di-lint.

## K-19 Aturan 3 diubah: tes unit per modul segera dijalankan (2026-10-11)
- Alasan: kode Fase 2-3 (~20 modul) belum pernah dijalankan; menunda ke Fase T menumpuk bug ke 17-19 Okt, dua hari sebelum
  Demo Day. Pengguna memilih menjalankan tes sekarang.
- Aturan baru: tiap modul Fase 1-5 ditulis bersama tb, lalu langsung dites sendiri (`make test-unit M=<modul>`) dan diperbaiki
  sampai hijau. Fase T tinggal integrasi dan regresi. Lint boleh kapan saja. Commit tetap ditunda sampai selesai dan
  dipecah per branch (fase1-2, fase3, fase4, fase5-6); Fase 5, T, 6 atas nama Faza Dzil.
- Simulator: Verilator 5.020 (apt) terlalu lama untuk cocotb 2.1 (`VerilatedVpi::clearEvalNeeded` tidak ada); tes unit memakai
  **Icarus 12** (`SIM=icarus`, default Makefile). `sva_stream.sv` (properti SVA) dilewati pada Icarus. Parameter string `.mem` dibuat
  tak bertipe (`parameter INIT_FILE = "..."`) karena Icarus 12 gagal mengikat `parameter string`; jalur `.mem` relatif terhadap ROOT.
- Hasil tes unit pertama (2026-10-11): quant, log_q, power, window, mel, fft512, feat_buf, dsp_chain, preemph, i2s_rx,
  frame_buf, cough_det semuanya LULUS bit-exact terhadap vektor golden. Dua perbaikan hanya di sisi tes (model mikrofon I2S tidak
  memulai slot kiri pertama; indeks di tes log_q/quant melebihi 6 bit). Lint Verilator: tanpa galat, hanya peringatan lebar/tak-terpakai.

## K-20 Fase 4: CNN output-stationary ditulis dan lulus tes unit (2026-10-11)
- Modul: `rtl/cnn/{cnn_cfg_pkg,pe_array,requant,gap,cnn_ctrl,cnn_top}.sv`. Bentuk layer/alamat ROM dari
  `golden/gen_cnn_rom.py` (sumber tunggal `params.layer_descs()`, P dari params); ROM dikemas P lane (`cnn_wrom/bias/rq.mem`).
- Pool pada akumulator sebelum requant (bit-exact dengan golden); satu unit requant berurutan; GAP unit sendiri; FC memakai
  pe_array yang sama; deskriptor sebagai fungsi paket (bukan array parameter, kompatibel Icarus).
- HASIL SIMULASI (Icarus, bukan laporan alat): logit cnn_top == golden bit-exact untuk 5 kasus sintetis (bobot acak),
  8 klip dengan bobot terlatih model_v2, dan P = 8 / 16 / 32.
- Siklus inferensi terukur di simulasi (tanpa pemuatan 2.000 siklus): P=8: 451.851; P=16: 226.490; P=32: 122.843.
  Waktu @98,304 MHz = siklus x 10,17 ns -> P=16: ~2,30 ms (BERGANTUNG Fmax; Fmax belum diukur, Fase 6).
- Catatan: sebagian besar tidak dioptimalkan (tulis requant tidak tumpang-tindih dengan MAC), sumber daya belum diukur.

## K-21 Target akurasi diganti pelaporan terukur + pembanding WHO TPP (2026-10-11)
- Alasan: 0,70 (batuk saja) dan 0,80 (fusi) adalah tebakan awal di CLAUDE.md, bukan syarat lomba atau standar medis. Hasil
  terukur: audio 0,52-0,63 (tidak beda dari nol permutasi); klinis 14 fitur 0,772 pada 1.828 peserta (205 positif); tambahan audio
  di atas klinis +0,003 (CI -0,021..+0,026) pada 360 peserta berlaudio; lintas-perangkat dan lintas-dataset 0,50-0,63.
- Standar pembanding: WHO TPP triase komunitas (sens >= 90%, spes >= 70%). Skor klinis kita: spes 38% pada sens 90%, sens 71% pada
  spes 70%; butuh AUROC ~0,90 (aproksimasi binormal). TIDAK dipenuhi dan tidak diklaim.
- Aturan baru: akurasi dilaporkan sebagai AUROC + CI 95% bootstrap atas peserta + nol permutasi + kontrol jenis kelamin/usia +
  hasil lintas-dataset/perangkat; setiap klaim melewati 0,5 harus menyebut apakah CI mencakup 0,5 atau nol permutasi.
  Target 0,70/0,80 dihapus sebagai syarat lulus. Penurunan kuantisasi <= 0,02 dan HW vs golden bit-exact tetap.
- Posisi alat: prioritas rujukan, bukan pengganti tes. Penguatan data (mis. CODA, K-15) = pasca-hackathon.
