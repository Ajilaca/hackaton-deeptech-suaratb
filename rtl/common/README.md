# rtl/common

| Berkas | Peran |
|---|---|
| `suaratb_pkg.sv` | Paket parameter (dihasilkan `golden/params.py`; jangan diedit) |
| `rom_sync.sv` | ROM sinkron 1 siklus, isi dari `$readmemh` |
| `ram_sdp.sv` | RAM dua-port sederhana (1 tulis, 1 baca sinkron) |
| `sync_fifo.sv` | FIFO sinkron valid/ready (DEPTH pangkat dua) |
| `sva_stream.sv` | Pemeriksa protokol stream untuk simulasi |
