"""Langkah 3 (uji murah): apakah embedding VGGish pralatih membawa sinyal TB yang tidak hanya jenis kelamin/perangkat?

    python -m model.embed_test embed          # hitung embedding 128-d per klip (Zambia dan TBscreen)
    python -m model.embed_test eval           # LR di atas embedding peserta, CV + residu sex/usia + lintas-dataset

Guru = VGGish (bobot dari TBscreen_code.zip, Zenodo 10431329, CC BY 4.0). Masukan: segmen 8240 sampel (16 kHz) diberi
nol sampai 15.600 sampel (96 frame = 0,96 s). Keluaran = 128-d (setelah ReLU, varian PyTorch TBscreen). Bobot hanya dimuat
dengan torch.load(weights_only=True). Embedding disimpan di $SUARATB_DATA/features/emb_<set>.npy; tidak di-commit.
Keputusan: bila AUROC embedding setelah residu jenis kelamin+usia tidak jelas di atas fitur log-mel (≈0,58), distilasi
tidak dilanjutkan.
"""
from __future__ import annotations
import importlib.util
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from golden import paths

CODE = paths.data_root() / "raw_tbscreen" / "code_x" / "TBscreen_code" / "utils"
NEED = 15600


class VGGish(nn.Module):
    def __init__(self):
        super().__init__()
        c = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True)]  # noqa: E731
        self.features = nn.Sequential(*c(1, 64), nn.MaxPool2d(2, 2), *c(64, 128), nn.MaxPool2d(2, 2),
                                      *c(128, 256), *c(256, 256), nn.MaxPool2d(2, 2),
                                      *c(256, 512), *c(512, 512), nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(512 * 24, 4096), nn.ReLU(True), nn.Linear(4096, 4096), nn.ReLU(True),
                                nn.Linear(4096, 128), nn.ReLU(True))

    def forward(self, x):
        x = self.features(x).permute(0, 2, 3, 1).contiguous()
        return self.fc(x.view(x.size(0), -1))


def _mel_module():
    spec = importlib.util.spec_from_file_location("mel_features", CODE / "audioset" / "mel_features.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def logmel(mf, seg: np.ndarray) -> np.ndarray:
    x = np.zeros(NEED, dtype=np.float64)
    x[:len(seg)] = seg[:NEED] / 32768.0
    lm = mf.log_mel_spectrogram(x, audio_sample_rate=16000, log_offset=0.01, window_length_secs=0.025,
                                hop_length_secs=0.010, num_mel_bins=64, lower_edge_hertz=125, upper_edge_hertz=7500)
    return lm[:96].astype(np.float32)


def embed(clips: np.ndarray, net: VGGish, mf, bs: int = 64) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(clips), bs):
            X = np.stack([logmel(mf, c) for c in clips[i:i + bs]])
            out.append(net(torch.from_numpy(X)[:, None]).numpy())
            if (i // bs) % 20 == 0:
                print(i, "/", len(clips), flush=True)
    return np.concatenate(out)


def zambia_clips(tag="v2"):
    fd = paths.features_dir()
    meta = pd.read_csv(fd / f"meta_{tag}.csv")
    ci = pd.read_csv(paths.processed_dir() / "clip_index.csv").set_index("path")
    meta["k"] = meta.groupby("path").cumcount()
    out, cache = [], {}
    for r in meta.itertuples():
        if r.path not in cache:
            cache.clear(); cache[r.path] = np.load(ci.loc[r.path, "out"])["clips"]
        out.append(cache[r.path][r.k])
    return np.stack(out)


def cmd_embed():
    torch.set_num_threads(4)
    net = VGGish()
    net.load_state_dict(torch.load(CODE / "model" / "pytorch_vggish.pth", map_location="cpu", weights_only=True))
    net.eval()
    mf = _mel_module()
    fd = paths.features_dir()
    np.save(fd / "emb_tbs.npy", embed(np.load(fd / "feats_tbs.npz")["clips"], net, mf))
    np.save(fd / "emb_zambia.npy", embed(zambia_clips(), net, mf))
    print("selesai")


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        cmd_embed()
    else:
        from model.embed_eval import main
        main()
