# model/distillation: HeAR -> CNN golden (eksploratori)

Dijalankan sebagai percobaan, bukan klaim klinis (K-23). Distilasi tidak memakai label TB.

| Berkas | Peran |
|---|---|
| `distill_hear.py` | Tabel klip (fitur hardware + embedding guru), distilasi per lipatan, evaluasi, kontrol (jaringan acak, target teracak), jalan pintas |
| `distill_export.py` | Murid + probe terlipat menjadi FC 16->1, PTQ int8, uji golden, ekspor `.mem`/`qmodel.npz`, vektor uji RTL |

Hasil: murid AUROC 0,654 (guru 0,672); kuantisasi int8 turun 0,008; RTL `cnn_top` dan `suaratb_top` bit-exact (`docs/vault/model.md` §5e).
