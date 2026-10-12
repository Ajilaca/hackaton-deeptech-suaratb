# tb/vectors: vektor uji

Vektor **sintetis** (sinus, derau, impuls, batuk buatan) yang dihasilkan `golden/gen/gen_vectors.py`; tidak berisi audio nyata.
Format `.mem`: satu kata hex komplemen-dua per baris, baris `//` adalah komentar; lebar kata ada di `meta.json` tiap blok.

| Direktori | Blok |
|---|---|
| `i2s_rx`, `preemph`, `frame_buf`, `cough_det` | Front-end dan detektor |
| `window`, `fft512`, `power`, `mel`, `log_q`, `quant` | DSP |
| `frontend` | Rangkaian front-end lengkap |
| `cnn` | Bobot dan ROM CNN (acak terkalibrasi), masukan dan logit golden |
