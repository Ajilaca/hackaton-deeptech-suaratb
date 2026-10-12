# model/clinical_models: gejala klinis dan fusi

Fitur klinis memakai hanya gejala, anamnesis, dan antropometri (14 fitur; versi diperluas). Tidak dipakai: xpert, kultur,
smear, LAM, prediksi AI X-ray, `ground_truth_tb`, kode klinik.

| Berkas | Peran |
|---|---|
| `fusion.py` | Matriks fitur klinis, fusi skor batuk + gejala (regresi logistik), laporan |
| `clinical.py` | Fitur klinis diperluas, LR/GBM, fusi dengan skor audio bersarang tanpa kebocoran |
| `clinical_full.py` | Klinis pada semua 1.828 peserta dan manfaat tambahan audio pada peserta berlaudio |
