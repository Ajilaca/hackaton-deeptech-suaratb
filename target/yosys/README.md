# target/yosys

| Berkas | Peran |
|---|---|
| `synth.sh` | Sintesis `synth_xilinx -family xc7` setelah `sv2v` (Yosys 0,33 belum membaca `import` paket dan `return` fungsi); tanpa timing |

Angka Yosys tidak memuat timing dan tidak memprediksi Vivado; dipakai sebagai pembanding lintas-alat.
