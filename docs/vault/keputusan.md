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
