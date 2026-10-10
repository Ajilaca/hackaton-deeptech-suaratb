"""Indeks dataset Zambia CIDRZ (Kaggle googlehealthai/google-health-ai) -> tabel peserta x rekaman.

Tidak menyalin audio; hanya membaca metadata dan header WAV. Root data dari $SUARATB_DATA.
Label: kolom `ground_truth_tb` (pos/neg) dari GHAI_Final_Data_2023.csv. Split HARUS per peserta (barcode).
"""
from __future__ import annotations
import re
import pathlib
import pandas as pd
import soundfile as sf
from golden import paths

META = "Metadata and Codebook/Metadata and Codebook/GHAI_Final_Data_2023.csv"
BC = re.compile(r"(\d\d-\d{3}-\d{4})")


def device_of(folder: str) -> str:
    f = folder.lower()
    if f.startswith("audio-recorder"):
        return "recorder"
    m = re.search(r"phone ([abc])$", f.split("/")[0].strip())
    return f"phone{m.group(1)}" if m else "unknown"


def load_meta(root: pathlib.Path | None = None) -> pd.DataFrame:
    root = root or paths.raw_dir()
    d = pd.read_csv(root / META, low_memory=False)
    d["y"] = (d["ground_truth_tb"] == "pos").astype(int)
    return d


def index_audio(root: pathlib.Path | None = None, with_header: bool = True) -> pd.DataFrame:
    root = root or paths.raw_dir()
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() == ".csv" or p.suffix.lower() == ".json":
            continue
        rel = p.relative_to(root)
        top = rel.parts[0]
        m = BC.search(p.name)
        r = dict(path=str(rel), device=device_of(top), barcode=m.group(1) if m else None, name=p.name)
        if with_header:
            try:
                i = sf.info(str(p))
                r.update(sr=i.samplerate, ch=i.channels, dur=i.duration, subtype=i.subtype)
            except Exception as e:  # berkas tak terbaca / bukan WAV
                r.update(sr=None, ch=None, dur=None, subtype=f"ERR:{type(e).__name__}")
        rows.append(r)
    return pd.DataFrame(rows)


def joined(root: pathlib.Path | None = None) -> pd.DataFrame:
    a = index_audio(root)
    m = load_meta(root)[["barcode", "y", "facility_code", "age", "sex", "coughdur", "hiv_status",
                         "xpert_results", "culture_results", "consent_obtained"]]
    return a.merge(m, on="barcode", how="left")


if __name__ == "__main__":
    j = joined()
    print(j.groupby("device").agg(n=("path", "size"), peserta=("barcode", "nunique"),
                                  dur_h=("dur", lambda s: s.sum() / 3600)).to_string())
    print("tanpa label:", int(j["y"].isna().sum()), " tanpa barcode:", int(j["barcode"].isna().sum()))
    print(j.drop_duplicates("barcode")["y"].value_counts(dropna=False).to_dict())
