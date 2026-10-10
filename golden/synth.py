"""Audio SINTETIS untuk uji golden dan kalibrasi Fase 0 (bukan data klinis, bukan batuk asli).

Semua generator deterministik dari seed. Keluaran int16 pada 16 kHz.
"""
from __future__ import annotations
import numpy as np

FS = 16000


def _i16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


def noise(rng, n: int, sigma: float, tilt: float = 0.0) -> np.ndarray:
    """Derau Gaussian float; tilt>0 membuat spektrum condong ke frekuensi rendah (mirip pink)."""
    w = rng.standard_normal(n)
    if tilt:
        W = np.fft.rfft(w)
        f = np.maximum(np.arange(len(W)), 1.0)
        w = np.fft.irfft(W / f ** tilt, n)
    w = w / (np.std(w) + 1e-12)
    return sigma * w


def _shape(rng, n: int, centers=((600, 300), (1500, 500), (3000, 900))) -> np.ndarray:
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / FS)
    g = sum(np.exp(-0.5 * ((f - c) / s) ** 2) for c, s in centers) + 0.05
    y = np.fft.irfft(W * g, n)
    return y / (np.std(y) + 1e-12)


def cough_burst(rng, rms: float, n_bursts: int = 3) -> np.ndarray:
    """Letupan mirip batuk: 1-3 burst derau terbentuk-formant, selubung attack cepat + peluruhan eksp."""
    total = int(0.55 * FS)
    y = np.zeros(total)
    t0 = 0
    amp = 1.0
    for _ in range(n_bursts):
        n = int(rng.uniform(0.05, 0.11) * FS)
        e = np.exp(-np.arange(n) / (rng.uniform(0.02, 0.05) * FS))
        e[: int(0.004 * FS)] *= np.linspace(0, 1, int(0.004 * FS))
        seg = _shape(rng, n) * e * amp
        y[t0: t0 + n] += seg[: total - t0]
        t0 += int(rng.uniform(0.11, 0.2) * FS)
        amp *= rng.uniform(0.4, 0.7)
        if t0 >= total - 800:
            break
    act = np.abs(y) > 0.1 * np.max(np.abs(y))
    return y * (rms / (np.sqrt(np.mean(y[act] ** 2)) + 1e-12))


def speech_like(rng, n: int, rms: float) -> np.ndarray:
    """Bukan batuk: harmonik f0~120-220 Hz + modulasi amplitudo ~4 Hz + sedikit desis."""
    t = np.arange(n) / FS
    f0 = rng.uniform(120, 220)
    y = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28)) / h for h in range(1, 12))
    y = y * (0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6.28)))
    y = y + 0.05 * rng.standard_normal(n)
    return y * (rms / (np.std(y) + 1e-12))


def click(rng, n: int, amp: float, pos: int | None = None) -> np.ndarray:
    y = np.zeros(n)
    p = rng.integers(n // 4, 3 * n // 4) if pos is None else pos
    y[p: p + 24] = amp * np.exp(-np.arange(24) / 6.0) * rng.choice([-1, 1])
    return y


def tone(n: int, f: float, amp: float) -> np.ndarray:
    return amp * np.sin(2 * np.pi * f * np.arange(n) / FS)


def stream(seed: int, seconds: float, bg_sigma: float, n_cough: int, snr_db: float,
           distract: tuple[str, ...] = (), level_step: float = 1.0):
    """Aliran panjang. Mengembalikan (x int16, daftar posisi sampel awal batuk).
    snr_db: RMS burst terhadap RMS derau latar. level_step: pengali derau latar setelah tengah aliran."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    bg = noise(rng, n, bg_sigma, tilt=0.5)
    if level_step != 1.0:
        bg[n // 2:] *= level_step
    x = bg.copy()
    pos = []
    slots = np.linspace(1.0, seconds - 1.2, max(n_cough, 1))
    for s in slots[:n_cough]:
        p = int((s + rng.uniform(-0.2, 0.2)) * FS)
        p = max(p, 8000)
        b = cough_burst(rng, bg_sigma * 10 ** (snr_db / 20.0))
        x[p: p + len(b)] += b[: n - p]
        pos.append(p)
    for d in distract:
        p = int(rng.uniform(1.0, seconds - 1.2) * FS)
        if d == "speech":
            m = int(0.8 * FS)
            x[p: p + m] += speech_like(rng, m, bg_sigma * 10 ** (snr_db / 20.0))
        elif d == "click":
            x += click(rng, n, 20 * bg_sigma * 10 ** (snr_db / 20.0), pos=p)
        elif d == "tone":
            m = int(1.0 * FS)
            x[p: p + m] += tone(m, rng.uniform(300, 2000), bg_sigma * 10 ** (snr_db / 20.0))
    return _i16(x), pos


def clip(seed: int, bg_sigma: float = 10.0, snr_db: float = 15.0, kind: str = "cough") -> np.ndarray:
    """Klip 0,5 s (8000 sampel). kind: cough | noise | speech | sine | impulse | silence."""
    rng = np.random.default_rng(seed)
    n = 8000
    bg = noise(rng, n, bg_sigma, tilt=0.5)
    if kind == "cough":
        b = cough_burst(rng, bg_sigma * 10 ** (snr_db / 20.0))
        bg[400: 400 + min(len(b), n - 400)] += b[: n - 400]
    elif kind == "speech":
        bg += speech_like(rng, n, bg_sigma * 10 ** (snr_db / 20.0))
    elif kind == "sine":
        return _i16(tone(n, 1000.0, bg_sigma))
    elif kind == "impulse":
        y = np.zeros(n)
        y[1000] = bg_sigma
        return _i16(y)
    elif kind == "silence":
        return np.zeros(n, dtype=np.int16)
    return _i16(bg)
