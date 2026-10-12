# model/data: data dan fitur

| Berkas | Peran |
|---|---|
| `dataset.py` | Indeks dataset Zambia CIDRZ: peserta x rekaman, label `ground_truth_tb`, perangkat |
| `audio_io.py` | Muat WAV -> mono int16 16 kHz (resample), skala asli dipertahankan |
| `prep_audio.py` | Segmentasi batuk memakai detektor golden (sama dengan hardware) -> klip 8.240 sampel |
| `build_data.py` | Tabel klip berlabel, fitur golden (beberapa lebar FFT) dan float-ref, split per peserta |
| `clipq.py` | Fitur kualitas klip dan topeng "mirip batuk" (heuristik ditetapkan sebelum melihat label) |
| `describe_data.py` | Ringkasan dataset ke `docs/vault/data.md` |
| `tbscreen.py` | TBscreen (Nairobi): klip -> segmen 8.240 sampel -> fitur golden |
| `audit_data.py` | Audit cakupan data audio: berkas -> barcode -> metadata, pemulihan barcode salah ketik |
