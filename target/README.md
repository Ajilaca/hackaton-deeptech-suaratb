# target: kode spesifik perangkat

Satu-satunya tempat kode yang bergantung perangkat/alat. Referensi laporan: Artix-7 XC7A35T-1 (Vivado). Bukan janji hardware
demo; FPGA di lokasi acara disediakan panitia (belum diumumkan).

| Subdirektori | Isi |
|---|---|
| `artix7/` | Sintesis dan implementasi Vivado (mode OOC) dengan sapuan P |
| `yosys/` | Sintesis lintas-vendor Yosys sebagai pembanding |

Hasil dan batasannya: `docs/vault/sintesis.md`. Belum ada pin, MMCM, atau constraint I/O (menunggu papan).
