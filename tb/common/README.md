# tb/common: utilitas testbench

| Berkas | Peran |
|---|---|
| `cocotb_utils.py` | Clock, reset, driver dan monitor stream |
| `drive.py` | Pengemudi stream bertingkat untuk modul DSP (data + bendera per siklus) |
| `runner.py` | Pembangun dan penjalan cocotb (Verilator/Icarus), direktori build di `target/build` |
| `vec.py` | Pemuat vektor `.mem` dan pembanding bit-exact |
| `uart_model.py` | Model UART host (kirim byte, monitor keluaran) |
| `golden_model.py` | Model CNN acak terkalibrasi yang sama dengan `golden/gen/gen_vectors.py` |
