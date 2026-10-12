# model/evaluation: protokol dan uji

| Berkas | Peran |
|---|---|
| `protocol.py` | Protokol evaluasi tetap: semesta, lipatan, metrik (AUROC terstratifikasi jenis kelamin, Δ vs klinis) |
| `permtest.py` | Uji permutasi label dan Δ, statistik maksimum atas kandidat; kriteria lolos praregistrasi di docstring |
| `device_eval.py` | Lintas-perangkat (perekam, Phone A/B/C) dengan statistik log-mel |
| `tbs_eval.py` | TBscreen dalam-dataset dan lintas-dataset Zambia <-> TBscreen |
| `probe_shortcut.py` | Apakah skor embedding hanya jalan pintas (jenis kelamin, klinik, perangkat) |
| `loco.py` | Leave-one-clinic-out terbatas (Kan <-> Cha+Chai) |
