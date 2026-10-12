# model/pretrained: embedding model pralatih

Bobot HeAR (Hugging Face, lisensi Health AI Developer Foundations) dan OPERA (GitHub, MIT) tidak di-commit; disimpan di
`$SUARATB_DATA/{hf,ext}`. Periksa lisensi sebelum mempublikasikan bobot atau embedding.

| Berkas | Peran |
|---|---|
| `embed_test.py` | Embedding VGGish (128-d). Catatan: jendela 0,96 s berisi ~47% nol, uji ini tidak sah untuk menyimpulkan |
| `embed_eval.py` | Evaluasi VGGish vs statistik log-mel, dalam-dataset dan lintas-dataset |
| `embed_pretrained.py` | Ekstraksi HeAR (jendela 2 s di sekitar onset) dan OPERA-CE/CT, evaluasi dengan protokol tetap |
| `cross_device_hear.py` | HeAR lintas-perangkat pada peserta yang sama: latih perangkat X, uji perangkat Y |
