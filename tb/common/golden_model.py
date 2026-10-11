"""Model CNN acak terkalibrasi yang sama dengan golden/gen_vectors.py (bobot ROM tb/vectors/cnn)."""
from golden.blocks import cnn as C
from golden.blocks import frontend as F
from golden.gen_vectors import test_signals


def random_calibrated_model():
    model = C.random_model(seed=123)
    feats = [F.to_cnn_input(F.features_q(x)) for x in test_signals().values()]
    C.calibrate_requant(model, feats)
    return model
