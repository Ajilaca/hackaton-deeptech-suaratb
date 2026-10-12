"""E1: audit cakupan data audio Zambia (berkas -> barcode -> metadata berlabel) dan pemulihan barcode salah ketik.

    python -m model.data.audit_data [--out $SUARATB_DATA/runs/reports/audit-data.md]

Aturan pemulihan (urut, tiap tahap diberi label tingkat keyakinan):
  1. normalisasi pola: pemisah '_', ' ', tanpa tanda hubung ('027399-0735', '02-39970740')    -> tier "pola"
  2. segmen tengah bukan 399 ('01-392-0905', '02-699-0731') diganti 399, bila ada di metadata -> tier "tengah"
  3. tidak ada di metadata: cocok pada 4 digit terakhir DAN klinik sama dengan folder audio,
     hanya bila kandidatnya tunggal                                                            -> tier "ekor+klinik" (LEMAH)
Berkas bernama timestamp ('1663841310.wav') tidak dapat dipulihkan dari nama dan dilaporkan apa adanya.
Tidak menyalin audio, tidak mengubah berkas; keluaran: barcode_fix.csv di processed/ dan laporan .md.
"""
from __future__ import annotations
import argparse
import pathlib
import re
import pandas as pd
from golden import paths
from model.data import dataset

FAC_OF_FOLDER = {"kanyama": "Kan", "chawama": "Cha", "chainda": "Chai"}
LOOSE = re.compile(r"(?<!\d)(\d{2})\D?(\d{3})\D?(\d{3,4})(?!\d)")


def folder_fac(path: str) -> str | None:
    p = path.lower()
    for k, v in FAC_OF_FOLDER.items():
        if k in p:
            return v
    return None


def fix_rows(a: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    mb = set(m.barcode)
    fac = m.set_index("barcode").facility_code.to_dict()
    tails: dict[str, list[str]] = {}
    for b in m.barcode:
        tails.setdefault(b[-4:], []).append(b)
    rows = []
    for r in a.itertuples():
        raw = r.barcode if isinstance(r.barcode, str) else None
        name = pathlib.Path(r.path).stem
        b, tier = None, "tanpa-barcode"
        if raw is not None and raw in mb:
            b, tier = raw, "asli"
        else:
            cand = raw
            if cand is None:
                mm = LOOSE.search(name)
                if mm and not re.fullmatch(r"\d{9,}", name.split(" ")[0]):
                    cand = f"{mm.group(1)}-{mm.group(2)}-{int(mm.group(3)):04d}"
                    tier = "pola"
            if cand and cand in mb:
                b = cand
                tier = "pola" if tier == "pola" else tier
            elif cand:
                c2 = cand[:3] + "399" + cand[6:]
                if c2 in mb:
                    b, tier = c2, "tengah"
                else:
                    ff = folder_fac(r.path)
                    ks = [x for x in tails.get(cand[-4:], []) if fac.get(x) == ff]
                    if len(ks) == 1:
                        b, tier = ks[0], "ekor+klinik"
                    else:
                        tier = "tak-terpulihkan"
        rows.append((r.path, r.device, raw, b, tier))
    return pd.DataFrame(rows, columns=["path", "device", "barcode_raw", "barcode_fix", "tier"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.runs_dir() / "reports" / "audit-data.md"))
    a_ = ap.parse_args()
    a = dataset.index_audio(with_header=False)
    a = a[a.name.str.lower().str.endswith((".wav", ".zip"))]
    m = dataset.load_meta().drop_duplicates("barcode")
    lab = m[m.ground_truth_tb.isin(["pos", "neg"])]
    f = fix_rows(a, m)
    f.to_csv(paths.processed_dir() / "barcode_fix.csv", index=False)
    yb = dict(zip(lab.barcode, lab.y))
    o = ["# Audit data audio Zambia (E1)", "",
         f"Metadata: {len(m)} peserta ({len(lab)} berlabel, {int(lab.y.sum())} positif). Deskripsi Kaggle menyebut audio dari 664 "
         "peserta; yang ada di disk dihitung di bawah.", "",
         "## Berkas -> barcode", "| tier | berkas |", "|---|---|"]
    for t, n in f.tier.value_counts().items():
        o.append(f"| {t} | {n} |")
    # peserta
    def cover(sel_tiers, label):
        s = f[f.tier.isin(sel_tiers)]
        bs = set(s.barcode_fix.dropna())
        pos = sum(yb.get(b, 0) for b in bs if b in yb)
        n = len([b for b in bs if b in yb])
        return label, n, int(pos)
    o += ["", "## Peserta berlabel dengan audio (semua perangkat)", "| cakupan | peserta | positif |", "|---|---|---|"]
    for lbl, tiers in (("barcode asli saja", ["asli"]), ("+ pola dan segmen tengah (yakin)", ["asli", "pola", "tengah"]),
                       ("+ ekor+klinik (lemah)", ["asli", "pola", "tengah", "ekor+klinik"])):
        l, n, p = cover(tiers, lbl)
        o.append(f"| {l} | {n} | {p} |")
    o += ["", "## Per perangkat (peserta berlabel, cakupan yakin)", "| perangkat | peserta | positif | berkas tak-terpulihkan |",
          "|---|---|---|---|"]
    for d, g in f.groupby("device"):
        bs = {b for b in g[g.tier.isin(["asli", "pola", "tengah"])].barcode_fix.dropna() if b in yb}
        o.append(f"| {d} | {len(bs)} | {sum(yb[b] for b in bs)} | {int(g.tier.isin(['tak-terpulihkan', 'tanpa-barcode']).sum())} |")
    left = f[f.tier.isin(["tak-terpulihkan", "tanpa-barcode"])]
    o += ["", f"## Berkas yang tidak dapat dipetakan ({len(left)})",
          "Nama berupa timestamp (mis. 1663841310.wav) atau barcode tanpa pasangan metadata; tidak dipulihkan tanpa menebak.", ""]
    ts = left.path.str.contains(r"/\d{10}[^/]*\.wav$", regex=True)
    o.append(f"- bernama timestamp: {int(ts.sum())}; barcode tanpa metadata: {int((~ts).sum())}")
    weak = f[f.tier == "ekor+klinik"]
    o += ["", f"## Pemulihan lemah ({len(weak)} berkas; tidak dipakai kecuali dikonfirmasi)", ""]
    for r in weak.itertuples():
        o.append(f"- {r.barcode_raw} -> {r.barcode_fix} ({r.device})")
    kl = paths.processed_dir() / "kaggle_filelist.json"
    if kl.exists():
        import json
        names = json.loads(kl.read_text())
        aud = [n for n in names if n.lower().endswith((".wav", ".zip")) and not n.startswith(("Google_AI", "Metadata"))]
        loc = {str(pathlib.Path(r).as_posix()) for r in a.path}
        o += ["", "## Kelengkapan unduhan (daftar berkas Kaggle vs disk)",
              f"- Daftar Kaggle: {len(aud)} berkas audio; di disk: {len(a)}; di daftar tetapi tidak di disk: "
              f"{len([n for n in aud if n not in loc])}.",
              "- Jadi rilis Kaggle memang hanya memuat audio ini; angka 664 peserta pada deskripsi tidak ada di rilis."]
    out = pathlib.Path(a_.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
