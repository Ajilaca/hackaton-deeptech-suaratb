# sw: perangkat lunak host

Penampung untuk perangkat lunak di luar akselerator. **Belum berisi kode** pada saat ini; hasil terkait ada di `model/`.

| Subdirektori | Rencana |
|---|---|
| `fusion/` | Fusi skor audio dengan gejala klinis di host (saat ini diuji di `model/clinical_models`) |
| `bench/` | Benchmark akselerator vs CPU (Fase 8) |
| `host/` | Aplikasi host/demo (UART: tulis CSR, kirim dataset, baca skor) |
