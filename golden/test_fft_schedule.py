"""Model jadwal siklus fft512.sv (Python, bukan simulasi RTL): memeriksa pengalamatan in-place, alur 5 tahap, tulis a'/b'
pada t+5/t+6 dan jeda DRAIN antar-stage menghasilkan keluaran yang sama bit-per-bit dengan golden fft512.
Mencerminkan: RAM 1R1W read-first (baca mengembalikan data LAMA bila alamat sama), baca a pada t, b pada t+1,
tulis a' pada t+5 dan b' pada t+6, butterfly berikutnya tiap 2 siklus, DRAIN siklus kosong setelah tiap stage."""
import numpy as np
from . import params
from .blocks.fft512 import fft512, twiddle_q15

W = params.FFT_W
N = params.NFFT
MASK = sum(b << i for i, b in enumerate(params.FFT_STAGE_SHIFT))


def bitrev(v, bits=9):
    return int(format(v, f"0{bits}b")[::-1], 2)


def schedule_fft(x, drain=8):
    cos, sin = twiddle_q15()
    re = [0] * N
    im = [0] * N
    for n in range(N):
        re[bitrev(n)] = int(x[n])
    pending = []        # (cycle, addr, re, im) penulisan yang dijadwalkan
    cycle = 0
    lim = 1 << (W - 1)

    def apply_writes(upto):
        # tulis semua yang jatuh tempo pada siklus <= upto (urut siklus)
        nonlocal pending
        due = sorted([p for p in pending if p[0] <= upto], key=lambda p: p[0])
        pending = [p for p in pending if p[0] > upto]
        for _, a, r, i in due:
            re[a], im[a] = r, i

    for st in range(9):
        half = 1 << st
        ss = (MASK >> st) & 1
        sh = 15 + ss
        for j in range(N // 2):
            pos = j & (half - 1)
            ia = ((j >> st) << (st + 1)) | pos
            ib = ia | half
            tw = pos << (8 - st)
            t = cycle
            # baca a pada t, b pada t+1 (read-first: tulis yang jatuh tempo <= t-1 sudah terlihat)
            apply_writes(t - 1)
            ar, ai = re[ia], im[ia]
            apply_writes(t)
            br, bi = re[ib], im[ib]
            wr, wi = int(cos[tw]), -int(sin[tw])
            tr = br * wr - bi * wi
            ti = br * wi + bi * wr
            rnd = 1 << (sh - 1)
            o = [((ar << 15) + tr + rnd) >> sh, ((ar << 15) - tr + rnd) >> sh,
                 ((ai << 15) + ti + rnd) >> sh, ((ai << 15) - ti + rnd) >> sh]
            o = [max(-lim, min(lim - 1, v)) for v in o]
            pending.append((t + 5, ia, o[0], o[2]))
            pending.append((t + 6, ib, o[1], o[3]))
            cycle += 2
        cycle += drain
    apply_writes(cycle + 10)
    return np.array(re, dtype=np.int64), np.array(im, dtype=np.int64)


def test_schedule_matches_golden_random():
    rng = np.random.default_rng(3)
    for _ in range(3):
        x = rng.integers(-(1 << 22), 1 << 22, size=N)
        re, im = schedule_fft(x)
        gre, gim, _ = fft512(x, W=W)
        assert np.array_equal(re, gre) and np.array_equal(im, gim)


def test_schedule_full_scale_and_impulse():
    for x in (np.full(N, (1 << 23) - 1), np.where(np.arange(N) % 2 == 0, (1 << 23) - 1, -(1 << 23)),
              np.eye(1, N, 5, dtype=np.int64)[0] * 12345):
        x = np.asarray(x, dtype=np.int64)
        re, im = schedule_fft(x)
        gre, gim, _ = fft512(x, W=W)
        assert np.array_equal(re, gre) and np.array_equal(im, gim)


def test_drain_margin():
    """DRAIN=8 aman (RTL). Urutan butterfly (indeks rendah dulu) membuat bahkan DRAIN=0 kebetulan benar; jeda dipertahankan
    sebagai margin keselamatan terhadap perubahan urutan/latensi alur, jadi tes ini hanya memastikan DRAIN >= 5 benar."""
    rng = np.random.default_rng(5)
    x = rng.integers(-(1 << 22), 1 << 22, size=N)
    gre, gim, _ = fft512(x, W=W)
    for d in (5, 8, 12):
        re, im = schedule_fft(x, drain=d)
        assert np.array_equal(re, gre) and np.array_equal(im, gim)
