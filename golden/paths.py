"""Lokasi data. Root dari env SUARATB_DATA (awal /data/suaratb); berkas besar TIDAK disimpan di repo."""
from __future__ import annotations
import os
import pathlib

DEFAULT_ROOT = "/data/suaratb"


def data_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("SUARATB_DATA", DEFAULT_ROOT))


def raw_dir() -> pathlib.Path:
    return data_root() / "raw"


def processed_dir() -> pathlib.Path:
    return data_root() / "processed"


def features_dir() -> pathlib.Path:
    return data_root() / "features"


def runs_dir() -> pathlib.Path:
    return data_root() / "runs"


def build_dir() -> pathlib.Path:
    return data_root() / "build"
