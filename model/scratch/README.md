# model/scratch: CNN dilatih dari nol

Arsitektur sama dengan golden (Conv 16/32/32, GAP, FC 32->16->1). Hasilnya negatif untuk audio (AUROC ~0,56); dipertahankan
sebagai pembanding dan sebagai jalur ekspor int8.

| Berkas | Peran |
|---|---|
| `baseline.py` | CNN float dan uji sensitivitas lebar FFT, CV 5-lipat per peserta |
| `mil_train.py` | Pelatihan MIL tingkat peserta (rata-rata logit k batuk), augmentasi, early stopping, uji permutasi |
| `mil2.py` | Varian CNN MIL yang diperbaiki (kurangi derau, klip mirip-batuk, epoch tetap) |
| `remodel.py` | Statistik log-mel per peserta + regresi logistik dengan kontrol ketat |
| `export.py` | Float -> int8 (PTQ, QAT bila perlu) -> `QModel` golden, tabel requant, berkas `.mem` |
