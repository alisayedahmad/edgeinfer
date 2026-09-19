"""manual int8 quantization, the python twin of c_engine/src/quantize.c.

weights are per-channel symmetric int8, activations per-tensor asymmetric
int8, biases int32, and requantization is gemmlowp-style fixed point (q31
multiplier + shift). the integer forward pass here is bit-exact with the
c engine, so it is also the reference for c_engine/test/test_ops.c.
"""
import math

import numpy as np
import torch

INT8_MIN, INT8_MAX = -128, 127


def round_away(x):
    # round half away from zero, same as c roundf
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def bn_affine(bn):
    # inference-mode bn is a per-channel scale and shift
    var, mean = bn.running_var.double(), bn.running_mean.double()
    scale = bn.weight.double() / torch.sqrt(var + bn.eps)
    shift = bn.bias.double() - mean * scale
    return scale.detach().numpy(), shift.detach().numpy()


def float_layers(model):
    """per-layer float params in c engine layout.

    activations are hwc, so conv weights become [cout][kh][kw][cin],
    depthwise [kh][kw][c], pointwise [cout][cin]. every layer carries the
    raw conv weight, the bn scale/shift, and the bn-folded weight and bias.
    """
    out = []
    for name, (conv, bn, _) in model.features.named_children():
        w = conv.weight.detach().double().numpy()
        scale, shift = bn_affine(bn)
        kind = name.rstrip("0123456789")
        if kind == "dw":
            w = w[:, 0].transpose(1, 2, 0)
            folded = w * scale
        elif kind == "pw":
            w = w[:, :, 0, 0]
            folded = w * scale[:, None]
        else:
            w = w.transpose(0, 2, 3, 1)
            folded = w * scale[:, None, None, None]
        out.append({
            "name": name, "kind": kind, "w": w, "bn_scale": scale, "bn_shift": shift,
            "wf": folded, "bf": shift, "kernel": conv.kernel_size, "stride": conv.stride, "pad": conv.padding,
        })
    return out


@torch.no_grad()
def calibrate(model, x, bs=512):
    """min/max of every tensor the int8 graph quantizes, from fp32 runs on x.

    x is (n, 1, 49, 10) normalized features. returns {name: (lo, hi)} for
    "input", each layer output after relu, and "fc".
    """
    ranges = {"input": (x.min().item(), x.max().item())}

    def track(name):
        def hook(module, inputs, out):
            lo, hi = ranges.get(name, (math.inf, -math.inf))
            ranges[name] = (min(lo, out.min().item()), max(hi, out.max().item()))
        return hook

    mods = list(model.features.named_children()) + [("fc", model.fc)]
    handles = [m.register_forward_hook(track(name)) for name, m in mods]
    model.eval()
    for i in range(0, len(x), bs):
        model(x[i:i + bs])
    for h in handles:
        h.remove()
    return ranges


def act_params(lo, hi):
    # asymmetric int8, range always covers zero so zero is exact
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = (hi - lo) / 255.0 or 1.0
    zp = int(np.clip(round_away(INT8_MIN - lo / scale), INT8_MIN, INT8_MAX))
    return scale, zp


def quantize_weights(w, axis):
    # symmetric per output channel on axis, [-127, 127] so negation never overflows
    reduce = tuple(i for i in range(w.ndim) if i != axis % w.ndim)
    scale = np.abs(w).max(axis=reduce, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(round_away(w / scale), -127, 127).astype(np.int8)
    return q, scale.reshape(-1)


def quantize_multiplier(m):
    # real multiplier m as m0 * 2^(shift - 31), m0 in [2^30, 2^31)
    if m == 0:
        return 0, 0
    frac, shift = math.frexp(m)
    m0 = int(round_away(frac * (1 << 31)))
    if m0 == 1 << 31:
        m0, shift = m0 // 2, shift + 1
    return m0, shift


def srdhm(a, b):
    # saturating rounding doubling high mul on int32 values
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    ab = a * b
    x = ab + np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    out = np.where(x >= 0, x >> 31, -((-x) >> 31))
    return np.where((a == b) & (a == -(1 << 31)), (1 << 31) - 1, out)


def rdbpot(x, exp):
    # rounding divide by power of two, ties away from zero
    x, exp = np.asarray(x, dtype=np.int64), np.asarray(exp, dtype=np.int64)
    mask = (np.int64(1) << exp) - 1
    threshold = (mask >> 1) + (x < 0)
    return (x >> exp) + ((x & mask) > threshold)


def requant(acc, m0, shift):
    shift = np.asarray(shift, dtype=np.int64)
    left, right = np.maximum(shift, 0), np.maximum(-shift, 0)
    return rdbpot(srdhm(np.asarray(acc, dtype=np.int64) << left, m0), right)


def quantize_tensor(x, scale, zp):
    # divide in float32 like the c code, round in float64 where x + 0.5 is exact
    v = (np.asarray(x, dtype=np.float32) / np.float32(scale)).astype(np.float64)
    return np.clip(round_away(v) + zp, INT8_MIN, INT8_MAX).astype(np.int8)


def dequantize_tensor(q, scale, zp):
    return (q.astype(np.float32) - zp) * np.float32(scale)


def quantize_model(model, calib_x):
    """int8 params for the c engine from a trained model and calibration data.

    returns a dict with the input quant params, per-layer int8 weights,
    int32 biases and per-channel multipliers, and the fc layer. relu is
    folded into each layer's clamp range.
    """
    ranges = calibrate(model, calib_x)
    in_scale, in_zp = act_params(*ranges["input"])
    layers, prev = [], (in_scale, in_zp)
    for f in float_layers(model):
        w_q, w_scale = quantize_weights(f["wf"], axis=-1 if f["kind"] == "dw" else 0)
        out_scale, out_zp = act_params(*ranges[f["name"]])
        layers.append(dict(f, **_int8_params(f["bf"], w_q, w_scale, prev, (out_scale, out_zp), relu=True)))
        prev = (out_scale, out_zp)

    fc_w = model.fc.weight.detach().double().numpy()
    fc_b = model.fc.bias.detach().double().numpy()
    w_q, w_scale = quantize_weights(fc_w, axis=0)
    fc = dict(name="fc", kind="fc", **_int8_params(fc_b, w_q, w_scale, prev, act_params(*ranges["fc"]), relu=False))
    return {"in_scale": in_scale, "in_zp": in_zp, "layers": layers, "fc": fc, "ranges": ranges}


def _int8_params(bias, w_q, w_scale, act_in, act_out, relu):
    (in_scale, in_zp), (out_scale, out_zp) = act_in, act_out
    mults = [quantize_multiplier(in_scale * s / out_scale) for s in w_scale]
    return {
        "w_q": w_q, "w_scale": w_scale,
        "b_q": np.clip(round_away(bias / (in_scale * w_scale)), -(1 << 31), (1 << 31) - 1).astype(np.int32),
        "m0": np.array([m for m, _ in mults], dtype=np.int32),
        "shift": np.array([s for _, s in mults], dtype=np.int32),
        "in_zp": in_zp, "out_zp": out_zp, "out_scale": out_scale,
        "act_min": out_zp if relu else INT8_MIN, "act_max": INT8_MAX,
    }


def _finish(acc, p):
    out = requant(acc + p["b_q"], p["m0"], p["shift"]) + p["out_zp"]
    return np.clip(out, p["act_min"], p["act_max"]).astype(np.int8)


def conv_s8(x, p, kind, stride=(1, 1), pad=(0, 0)):
    """one int8 layer on an hwc tensor, bit-exact with the c kernels.

    padding uses the input zero point, which is the same as skipping padded
    taps since (zp - zp) * w contributes nothing.
    """
    x = x.astype(np.int64) - p["in_zp"]
    if kind == "pw":
        return _finish(x @ p["w_q"].astype(np.int64).T, p)
    x = np.pad(x, ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)))
    w = p["w_q"].astype(np.int64)
    kh, kw = (w.shape[0], w.shape[1]) if kind == "dw" else (w.shape[1], w.shape[2])
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(0, 1))[::stride[0], ::stride[1]]
    if kind == "dw":
        acc = np.einsum("hwcij,ijc->hwc", win, w)
    else:
        acc = np.einsum("hwcij,oijc->hwo", win, w)
    return _finish(acc, p)


def avgpool_s8(x):
    # global average of raw int8 values, output keeps the input quant params
    n = x.shape[0] * x.shape[1]
    acc = x.astype(np.int64).sum(axis=(0, 1))
    return np.where(acc > 0, (acc + n // 2) // n, -((n // 2 - acc) // n)).astype(np.int8)


def forward_int8(q, x):
    """int8 forward on one (49, 10, 1) int8 input.

    returns int8 logits and the int8 output of every layer, pool included.
    """
    taps = []
    for layer in q["layers"]:
        x = conv_s8(x, layer, layer["kind"], layer["stride"], layer["pad"])
        taps.append(x)
    x = avgpool_s8(x)
    taps.append(x)
    logits = conv_s8(x[None, None], q["fc"], "pw")[0, 0]
    return logits, taps
