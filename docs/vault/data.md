# Data (Google + CIDRZ Health AI Evaluation Zambia)

Dihasilkan `python -m model.describe_data`. Lokasi: `$SUARATB_DATA/raw` (= `/data/suaratb/raw`); tidak di-commit.

## Sumber dan lisensi
- Kaggle `googlehealthai/google-health-ai`, judul "Google + CIDRZ Health AI Evaluation Zambia".
- Lisensi menurut metadata Kaggle: **ODbL-1.0** (ODbL-1.0). Halaman dataset dan syarat tambahannya BELUM dibaca; verifikasi manual sebelum bobot/turunan dipublikasikan.
- Studi: validasi algoritma TB dari X-ray; audio batuk dari 664 peserta (menurut deskripsi) pada 4 perangkat serentak (perekam, Phone A/B/C) di 3 klinik Lusaka (Chainda South, Chawama, Kanyama).
- Aturan proyek: jangan redistribusi, jangan identifikasi ulang, split per peserta.

## Yang sudah diunduh (WAV)
| folder | berkas | peserta | jam | sample rate | kanal | GB |
|---|---|---|---|---|---|---|
| Audio-Recorder-Chainda-South | 36 | 35 | 0.5 | 44100 | 2 | 0.29 |
| Audio-Recorder-Chawama | 100 | 87 | 1.4 | 44100 | 2 | 0.91 |
| Audio-Recorder-Kanyama | 145 | 137 | 1.8 | 44100 | 2 | 1.15 |
| Chainda South Phone A | 35 | 35 | 0.4 | 192000 | 2 | 2.20 |
| Chawama Phone A | 100 | 88 | 1.3 | 192000 | 2 | 7.16 |
| Kanyama Phone A | 262 | 254 | 3.1 | 192000 | 2 | 16.89 |

Total berkas WAV: 678; tidak terbaca: 0; tanpa barcode di nama: 14.

## Metadata dan label
- `Metadata and Codebook/.../GHAI_Final_Data_2023.csv`: 1.828 peserta x 40 kolom (barcode, gejala, hasil xpert/culture, `ground_truth_tb` pos/neg, `facility_code`, prediksi AI X-ray, dst.). Kunci gabung audio: `barcode` = bagian nama berkas `NN-399-NNNN`.
- `ground_truth_tb`: 205 pos, 1.623 neg di seluruh 1.828 peserta.
- Audio berlabel: 620 dari 678 berkas WAV; peserta unik berlabel 360.

## Segmentasi (detektor golden, `model.prep_audio`)
- Rekaman diproses: 679; klip batuk: 8056; median klip/rekaman 11.
- Hash duplikat (rekaman identik): 28.

## Catatan kualitas
- Satu WAV = satu sesi (±40-70 detik di perekam) berisi banyak batuk; 21 peserta punya lebih dari satu rekaman.
- Beberapa berkas bernama `... (1).wav` (duplikat nama); unduhan via `kaggle -f` gagal untuk sebagian, lihat `scripts/dl_kaggle_phoneA.log`.
- Positif per klinik sangat tidak seimbang (Chainda South: 1 positif di perekam).
