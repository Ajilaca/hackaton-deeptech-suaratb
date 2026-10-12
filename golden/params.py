"""SuaraTB: satu-satunya sumber parameter (golden, tabel, vektor, budget, paket SV).

Status nilai: lihat CALIBRATION_SOURCE. Selama "synthetic", konstanta yang bergantung
pada data (MEL_SHIFT, FEAT_*, detektor, requant) adalah PROVISIONAL dan harus
dihitung ulang dengan `python -m golden.analysis.calibrate --data data/raw` begitu dataset ada.
Semua keputusan dan alasannya: docs/vault/keputusan.md.
"""
from __future__ import annotations

CALIBRATION_SOURCE = "synthetic"  # "synthetic" | "data/raw"

# ---- Audio / framing -------------------------------------------------------
FS = 16000
FRAME_LEN = 400            # 25 ms
HOP = 160                  # 10 ms
NFFT = 512
N_BINS = NFFT // 2 + 1     # 257
N_FRAMES = 50              # segmen fitur: 50 frame = PRE_ROLL + POST_ONSET
PRE_ROLL = 5
POST_ONSET = 45
# 50 frame butuh (50-1)*160+400 = 8240 sampel (0,515 s). Klip 0,5 s (8000 sampel)
# di mode dataset diberi nol 240 sampel di ekor (keputusan K-03).
CLIP_SAMPLES = 8000
CLIP_SAMPLES_PADDED = (N_FRAMES - 1) * HOP + FRAME_LEN

# ---- Pre-emphasis / window -------------------------------------------------
PREEMPH_Q15 = 31785        # round(0.97 * 2^15)
WIN_FRAC = 15              # Hamming Q1.15 (simetris, 400 titik, zero-pad ke 512)

# ---- FFT ---------------------------------------------------------------------
FFT_W = 24                 # lebar data (bit, bertanda). K-04: 18 bit gagal (SNR ~ -1 dB pada derau 3 LSB)
FFT_GUARD = FFT_W - 16     # bit pecahan tambahan di atas sampel 16 bit
PREEMPH_FRAC = FFT_GUARD   # K-08: pre-emphasis mengeluarkan 16+G bit (tanpa pembulatan ke int16)
TW_FRAC = 15               # twiddle Q1.15
FFT_STAGE_SHIFT = (1,) * 9  # skala 1/2 per stage (9 stage) -> total 1/512

# ---- Mel / log / kuantisasi ------------------------------------------------
N_MEL = 40
MEL_FMIN = 50.0
MEL_FMAX = 8000.0
MEL_W_FRAC = 12            # bobot mel unsigned Q0.12 (puncak dipotong ke 4095)
MEL_SHIFT = 0              # geser kanan akumulator mel sebelum log (K-05: 0 = tanpa kehilangan)
LOG_IN_BITS = 58           # masukan log2 (K-05): batas atas akumulator mel pada FFT_W=24, tanpa saturasi
LOG_LUT_BITS = 8           # 256 entri
LOG_FRAC = 8               # keluaran Q6.8 (6 bit bulat karena leading-one 0..57)
FEAT_OFFSET = 8265         # dari audio NYATA perekam (golden.analysis.calibrate_real, K-14); PROVISIONAL untuk INMP441
FEAT_SHIFT = 5             # q = sat8(round((L - FEAT_OFFSET) / 2^FEAT_SHIFT)); langkah 0,125 oktaf (0,38 dB)


# ---- Batas analitik (turunan; dipakai budget.py dan tes) ---------------------------
# Parseval: sum_{k<512} |X_k|^2 = sum|x w|^2 * 2^(2G) / 512 (FFT diskalakan 1/512)
POWER_BITS = 2 * FFT_W - 1                                     # P[k] tak bertanda
_P_SUM_MAX = FRAME_LEN * (2 ** 15) ** 2 * 4 ** FFT_GUARD // NFFT   # batas sum_k P[k]
MEL_ACC_BITS = (_P_SUM_MAX * ((1 << MEL_W_FRAC) - 1)).bit_length()  # batas akumulator mel

# ---- CNN -----------------------------------------------------------------------
C1, C2, C3 = 16, 32, 32    # kanal; turunkan bila FPGA kecil
FC_HIDDEN = 16
P = 16                     # PE paralel (8/16/32)
ACC_BITS = 24
ACT_BITS = 8
WGT_BITS = 8
BIAS_BITS = 24
REQ_M0_BITS = 16           # M0 Q15 (unsigned, 0..32767)
REQ_SHIFT_MAX = 31
LOGIT_BITS = 16            # keluaran FC akhir (int16, tanpa ReLU)
GAP_RECIP_FRAC = 15

# ---- Detektor batuk -------------------------------------------------------
DET_K = 8                  # trigger: E > DET_K * N   (K-07)
DET_EMA_SHIFT = 6          # N += (E - N) / 2^6
DET_MIN_HOPS = 2           # >= 2 hop berurutan di atas ambang
DET_HOLD_HOPS = 30         # 300 ms refractory setelah segmen selesai
DET_N_FLOOR = 640          # lantai N (satuan E): derau sigma ~2 LSB, 160 sampel (asumsi)
DET_E_SHIFT = 0            # E = sum(x^2) >> DET_E_SHIFT

# ---- Turunan -------------------------------------------------------------------


def layer_descs(c1: int = C1, c2: int = C2, c3: int = C3, hidden: int = FC_HIDDEN):
    """Deskriptor layer (sumber tunggal untuk golden, budget, dan nanti tabel deskriptor RTL)."""
    h, w = N_MEL, N_FRAMES
    d = []
    cin = 1
    for i, cout in enumerate((c1, c2, c3), start=1):
        d.append(dict(name=f"conv{i}", kind="conv", cin=cin, cout=cout, h=h, w=w, k=3, pool=True))
        cin, h, w = cout, h // 2, w // 2
    d.append(dict(name="fc1", kind="fc", cin=cin, cout=hidden, h=1, w=1, k=1, pool=False,
                  gap_hw=h * w))
    d.append(dict(name="fc2", kind="fc", cin=hidden, cout=1, h=1, w=1, k=1, pool=False))
    return d


def n_weights(**kw) -> int:
    return sum(l["cin"] * l["cout"] * l["k"] * l["k"] for l in layer_descs(**kw))


def n_biases(**kw) -> int:
    return sum(l["cout"] for l in layer_descs(**kw))


def n_macs(**kw) -> int:
    t = 0
    for l in layer_descs(**kw):
        t += l["h"] * l["w"] * l["cin"] * l["cout"] * l["k"] * l["k"]
    return t


def sv_package(name: str = "suaratb_pkg") -> str:
    """Paket SystemVerilog dari parameter di atas (dipakai RTL Fase 2+). Nama berawalan '_' dilewati;
    nilai di luar int 32 bit ditulis sebagai longint."""
    kv = [(k, v) for k, v in globals().items()
          if k.isupper() and not k.startswith("_") and isinstance(v, int) and not isinstance(v, bool)]
    lines = [f"// Dihasilkan oleh golden/params.py (sumber: {CALIBRATION_SOURCE}). JANGAN diedit manual.",
             f"package {name};"]
    for k, v in kv:
        ty = "int" if -(1 << 31) <= v < (1 << 31) else "longint"
        lines.append(f"  localparam {ty} {k} = {v};")
    lines.append(f"  localparam int FFT_STAGE_SHIFT_MASK = {sum(b << i for i, b in enumerate(FFT_STAGE_SHIFT))};")
    lines.append(f"  localparam int N_FFT_STAGES = {len(FFT_STAGE_SHIFT)};")
    lines.append("endpackage")
    return "\n".join(lines) + "\n"
