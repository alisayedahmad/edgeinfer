"""int8 checks: the arithmetic, the accuracy cost, and c against python."""
import subprocess

import numpy as np
import pytest
import torch

from runtime import profile
from train import quant
from train.ds_cnn import DSCNN, N_FRAMES, N_MFCC


@pytest.fixture(scope="module")
def quantized():
    torch.manual_seed(1)
    model = DSCNN(n_classes=12, width=16, blocks=2).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_var.uniform_(0.5, 2.0)
                module.weight.uniform_(0.5, 1.5)
    x = torch.randn(256, 1, N_FRAMES, N_MFCC)
    return model, x, quant.quantize_model(model, x[:128])


def test_round_trip_within_half_a_step():
    x = np.random.default_rng(0).uniform(-3, 3, 4096).astype(np.float32)
    scale, zp = 6.0 / 255, 5
    q = quant.quantize_tensor(x, scale, zp)
    back = quant.dequantize_tensor(q, scale, zp)
    inside = (x > (-128 - zp) * scale) & (x < (127 - zp) * scale)
    assert np.abs(back[inside] - x[inside]).max() <= scale / 2 + 1e-6


def test_requant_matches_the_real_multiplier():
    rng = np.random.default_rng(0)
    acc = rng.integers(-(1 << 22), 1 << 22, 4096)
    for real in [1e-4, 0.013, 0.37, 0.999, 1.0, 2.5]:
        m0, shift = quant.quantize_multiplier(real)
        got = quant.requant(acc, m0, shift)
        # double rounding costs at most one step
        assert np.abs(got - acc * real).max() <= 1.0


def test_zero_is_exact():
    # an asymmetric range must land real zero on an integer, padding depends on it
    for lo, hi in [(-3.0, 5.0), (0.0, 6.0), (-2.0, 0.0)]:
        scale, zp = quant.act_params(lo, hi)
        assert quant.dequantize_tensor(np.array([zp], np.int8), scale, zp)[0] == 0.0


def test_int8_forward_tracks_fp32(quantized):
    model, x, q = quantized
    with torch.no_grad():
        ref = model(x[128:]).numpy()
    agree, worst = 0, 0.0
    for i in range(64):
        sample = quant.quantize_tensor(x[128 + i, 0].numpy()[..., None], q["in_scale"], q["in_zp"])
        logits, _ = quant.forward_int8(q, sample)
        deq = quant.dequantize_tensor(logits, q["fc"]["out_scale"], q["fc"]["out_zp"])
        agree += deq.argmax() == ref[i].argmax()
        worst = max(worst, np.abs(deq - ref[i]).max())
    assert agree >= 60, f"argmax agreement {agree}/64"
    assert worst < 0.1 * (ref.max() - ref.min())


def test_weights_are_per_channel_symmetric(quantized):
    _, _, q = quantized
    for layer in q["layers"]:
        assert layer["w_q"].dtype == np.int8
        assert np.abs(layer["w_q"]).max() <= 127
        # one scale per output channel, not one for the whole tensor
        channels = layer["w_q"].shape[-1 if layer["kind"] == "dw" else 0]
        assert layer["w_scale"].shape == (channels,)


def test_relu_folds_into_the_clamp(quantized):
    _, _, q = quantized
    for layer in q["layers"]:
        assert layer["act_min"] == layer["out_zp"]
    assert q["fc"]["act_min"] == -128


def test_c_engine_matches_the_python_reference():
    """the c kernels and train/quant.py must agree bit for bit.

    needs the exported headers and a built cli, so it skips on a bare
    checkout: make c-engine builds both.
    """
    features = profile.ARTIFACTS / "test_features.f32"
    if not features.exists():
        pytest.skip("no exported features, run python -m train.export")
    try:
        cli = profile.c_cli()
    except SystemExit:
        pytest.skip("c engine cli not built")

    x, _ = profile.eval_set()
    samples = min(32, len(x))
    out = profile.ARTIFACTS / "test_int8_logits.f32"
    subprocess.run([str(cli), "eval", "int8", str(features), str(samples), str(out)],
                   check=True, capture_output=True)
    got = np.fromfile(out, np.float32).reshape(samples, -1)

    model, ckpt = _load_exported_model()
    q = quant.quantize_model(model, torch.from_numpy(profile.calib_set()))
    for i in range(samples):
        sample = quant.quantize_tensor(x[i, 0][..., None], q["in_scale"], q["in_zp"])
        logits, _ = quant.forward_int8(q, sample)
        deq = quant.dequantize_tensor(logits, q["fc"]["out_scale"], q["fc"]["out_zp"])
        assert np.array_equal(deq, got[i]), f"sample {i} differs"


def _load_exported_model():
    from train.ds_cnn import load

    ckpt = profile.REPO / "artifacts" / "ds_cnn.pt"
    if not ckpt.exists():
        pytest.skip("no checkpoint, train or export one first")
    return load(ckpt)
