"""Tulis ringkasan dataset ke docs/vault/data.md (dapat dijalankan ulang setelah unduhan bertambah).

    python -m model.describe_data
"""
from __future__ import annotations
import json
import pathlib
import pandas as pd
from golden import paths
from model import dataset


def main(out="docs/vault/data.md"):
    root = paths.raw_dir()
    meta = json.loads((root / "dataset-metadata.json").read_text())["info"]
    a = dataset.index_audio(root)
    wav = a[a.name.str.lower().str.endswith(".wav")]
    j = wav.merge(dataset.load_meta(root)[["barcode", "y", "facility_code"]], on="barcode", how="left")
    top = wav.path.str.split("/").str[0]
    g = wav.assign(top=top).groupby("top").agg(berkas=("path", "size"), peserta=("barcode", "nunique"),
                                              jam=("dur", lambda s: s.sum() / 3600),
                                              sr=("sr", lambda s: ",".join(sorted({str(int(x)) for x in s.dropna()}))),
                                              ch=("ch", lambda s: ",".join(sorted({str(int(x)) for x in s.dropna()}))))
    size = {t: sum(p.stat().st_size for p in (root / t).rglob("*") if p.is_file()) / 1e9 for t in g.index}
    g["GB"] = pd.Series(size)
    err = wav[wav["subtype"].astype(str).str.startswith("ERR")]
    ci = pd.read_csv(paths.processed_dir() / "clip_index.csv") if (paths.processed_dir() / "clip_index.csv").exists() else None
    o = ["# Data (Google + CIDRZ Health AI Evaluation Zambia)", "",
         "Dihasilkan `python -m model.describe_data`. Lokasi: `$SUARATB_DATA/raw` (= `/data/suaratb/raw`); tidak di-commit.", "",
         "## Sumber dan lisensi",
         f"- Kaggle `{meta['ownerUser']}/{meta['datasetSlug']}`, judul \"{meta['title']}\".",
         f"- Lisensi menurut metadata Kaggle: **{', '.join(l['name'] for l in meta['licenses'])}** (ODbL-1.0). "
         "Halaman dataset dan syarat tambahannya BELUM dibaca; verifikasi manual sebelum bobot/turunan dipublikasikan.",
         "- Studi: validasi algoritma TB dari X-ray; audio batuk dari 664 peserta (menurut deskripsi) pada 4 perangkat serentak "
         "(perekam, Phone A/B/C) di 3 klinik Lusaka (Chainda South, Chawama, Kanyama).",
         "- Aturan proyek: jangan redistribusi, jangan identifikasi ulang, split per peserta.", "",
         "## Yang sudah diunduh (WAV)",
         "| folder | berkas | peserta | jam | sample rate | kanal | GB |", "|---|---|---|---|---|---|---|"]
    for t, r in g.iterrows():
        o.append(f"| {t} | {int(r.berkas)} | {int(r.peserta)} | {r.jam:.1f} | {r.sr} | {r.ch} | {r.GB:.2f} |")
    o += ["", f"Total berkas WAV: {len(wav)}; tidak terbaca: {len(err)}; tanpa barcode di nama: {int(wav.barcode.isna().sum())}.", "",
          "## Metadata dan label",
          "- `Metadata and Codebook/.../GHAI_Final_Data_2023.csv`: 1.828 peserta x 40 kolom (barcode, gejala, hasil xpert/culture, "
          "`ground_truth_tb` pos/neg, `facility_code`, prediksi AI X-ray, dst.). Kunci gabung audio: `barcode` = bagian nama berkas `NN-399-NNNN`.",
          "- `ground_truth_tb`: 205 pos, 1.623 neg di seluruh 1.828 peserta.",
          f"- Audio berlabel: {int(j.y.notna().sum())} dari {len(j)} berkas WAV; peserta unik berlabel {j[j.y.notna()].barcode.nunique()}.", ""]
    if ci is not None:
        ok = ci[ci.status.isin(['ok', 'cached'])]
        o += ["## Segmentasi (detektor golden, `model.prep_audio`)",
              f"- Rekaman diproses: {len(ci)}; klip batuk: {int(ok.n_clips.sum())}; median klip/rekaman {ok.n_clips.median():.0f}.",
              f"- Hash duplikat (rekaman identik): {int(ok.duplicated('hash').sum())}.", ""]
    o += ["## Catatan kualitas",
          "- Satu WAV = satu sesi (±40-70 detik di perekam) berisi banyak batuk; 21 peserta punya lebih dari satu rekaman.",
          "- Beberapa berkas bernama `... (1).wav` (duplikat nama); unduhan via `kaggle -f` gagal untuk sebagian, lihat `scripts/dl_kaggle_phoneA.log`.",
          "- Positif per klinik sangat tidak seimbang (Chainda South: 1 positif di perekam)."]
    pathlib.Path(out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
