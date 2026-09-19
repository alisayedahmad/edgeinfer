"""write c_engine/test/vectors.h for test_ops.c.

fp32 references come from pytorch, int8 references from train/quant.py
(bit-exact with the c kernels), the mfcc reference from
data/speech_commands.py. weights are random, so the tests need no model.

    python -m c_engine.test.gen_vectors
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import speech_commands as sc
from train import quant
from train.export import c_array, c_header

OUT = Path(__file__).resolve().parent / "vectors.h"


def hwc(t):
    # (1, c, h, w) torch -> hwc numpy
    return t[0].permute(1, 2, 0).numpy()


def fp32_vectors(rng):
    def randn(*shape, scale=1.0):
        return torch.from_numpy((rng.standard_normal(shape) * scale).astype(np.float32))

    v = {}
    x = randn(1, 1, 49, 10)
    w, b = randn(8, 1, 10, 4, scale=0.3), randn(8)
    v.update(conv_x=hwc(x), conv_w=w.permute(0, 2, 3, 1).numpy(), conv_b=b)
    v["conv_y"] = hwc(F.conv2d(x, w, b, stride=2, padding=(5, 1)))

    x = randn(1, 12, 25, 5)
    w, b = randn(16, 12, 1, 1, scale=0.3), randn(16)
    v.update(act_x=hwc(x), pw_w=w[:, :, 0, 0].numpy(), pw_b=b, pw_y=hwc(F.conv2d(x, w, b)))
    w, b = randn(12, 1, 3, 3, scale=0.3), randn(12)
    v.update(dw_w=w[:, 0].permute(1, 2, 0).numpy(), dw_b=b, dw_y=hwc(F.conv2d(x, w, b, padding=1, groups=12)))

    # conv -> bn -> relu in pytorch vs the folded kernels in c, folded the way
    # train/quant.py folds the real model
    gamma, beta = rng.uniform(0.5, 1.5, 12), rng.standard_normal(12)
    mean, var = rng.standard_normal(12) * 0.5, rng.uniform(0.5, 2.0, 12)
    scale = gamma / np.sqrt(var + 1e-5)
    shift = beta - mean * scale
    bn = [torch.from_numpy(a.astype(np.float32)) for a in (mean, var, gamma, beta)]
    v.update(bn_scale=scale, bn_shift=shift, bn_y=hwc(F.batch_norm(x, *bn, eps=1e-5)))
    v["relu_y"] = hwc(F.relu(x))
    v["fused_dw_w"] = w[:, 0].permute(1, 2, 0).double().numpy() * scale
    v["fused_dw_y"] = hwc(F.relu(F.batch_norm(F.conv2d(x, w, None, padding=1, groups=12), *bn, eps=1e-5)))
    w = randn(12, 12, 1, 1, scale=0.3)
    v["fused_pw_raw_w"] = w[:, :, 0, 0].numpy()
    v["fused_pw_w"] = w[:, :, 0, 0].double().numpy() * scale[:, None]
    v["fused_pw_y"] = hwc(F.relu(F.batch_norm(F.conv2d(x, w), *bn, eps=1e-5)))

    pooled = F.adaptive_avg_pool2d(x, 1).flatten()
    w, b = randn(35, 12), randn(35)
    v.update(pool_y=pooled.numpy(), fc_w=w.numpy(), fc_b=b, fc_y=F.linear(pooled, w, b).numpy())
    return [c_array("float", name, a) for name, a in v.items()]


def int8_layer(rng, name, w_shape, axis, relu=True):
    # random but realistic int8 params: per-channel weights, requant from real scales
    w_q, w_scale = quant.quantize_weights(rng.standard_normal(w_shape) * 0.2, axis)
    bias = rng.standard_normal(w_q.shape[axis]) * 0.5
    act_in = (0.05, int(rng.integers(-20, 20)))
    act_out = (0.08, -128 if relu else 3)
    p = quant.int8_params(bias, w_q, w_scale, act_in, act_out, relu)
    code = [
        c_array("int8_t", f"{name}_w", p["w_q"]), c_array("int32_t", f"{name}_bias", p["b_q"]),
        c_array("int32_t", f"{name}_mult", p["m0"]), c_array("int8_t", f"{name}_shift", p["shift"]),
        f"static const layer_q_t {name}_layer = {{{name}_w, {name}_bias, {name}_mult, {name}_shift, "
        f"{act_in[1]}, {act_out[1]}, {p['act_min']}, {p['act_max']}, {act_out[0]}f}};\n",
    ]
    return p, code


def int8_vectors(rng):
    code = []
    x = rng.integers(-128, 128, (49, 10, 1)).astype(np.int8)
    p, c = int8_layer(rng, "q_conv", (8, 10, 4, 1), 0)
    code += c + [c_array("int8_t", "q_conv_x", x),
                 c_array("int8_t", "q_conv_y", quant.conv_s8(x, p, "conv", (2, 2), (5, 1)))]

    x = rng.integers(-128, 128, (25, 5, 12)).astype(np.int8)
    code.append(c_array("int8_t", "q_x", x))
    p, c = int8_layer(rng, "q_dw", (3, 3, 12), -1)
    code += c + [c_array("int8_t", "q_dw_y", quant.conv_s8(x, p, "dw", (1, 1), (1, 1)))]
    p, c = int8_layer(rng, "q_pw", (16, 12), 0)
    code += c + [c_array("int8_t", "q_pw_y", quant.conv_s8(x, p, "pw"))]
    pooled = quant.avgpool_s8(x)
    p, c = int8_layer(rng, "q_fc", (35, 12), 0, relu=False)
    code += c + [c_array("int8_t", "q_pool_y", pooled),
                 c_array("int8_t", "q_fc_y", quant.conv_s8(pooled[None, None], p, "pw")[0, 0])]

    # exact halves and a clamp exercise the rounding
    x = rng.standard_normal(490).astype(np.float32) * 3
    x[:4] = [0.5 * 0.0625, -0.5 * 0.0625, 1.5 * 0.0625, 100.0]
    code += [c_array("float", "quant_x", x), c_array("int8_t", "quant_y", quant.quantize_tensor(x, 0.0625, 5))]

    # multipliers >= 1 take the left-shift path
    acc = rng.integers(-(1 << 24), 1 << 24, 256)
    real = rng.uniform(1e-5, 0.99, 256)
    real[:3] = [1.0, 1.7, 3.2]
    m0, shift = np.array([quant.quantize_multiplier(r) for r in real]).T
    code += [c_array("int32_t", "rq_acc", acc), c_array("int32_t", "rq_mult", m0),
             c_array("int8_t", "rq_shift", shift), c_array("int32_t", "rq_y", quant.requant(acc, m0, shift))]
    return code


def mfcc_vectors(rng):
    # a chirp with noise, a quiet stretch, digital silence and a clipped burst
    t = np.arange(sc.N_SAMPLES) / sc.SR
    y = 0.6 * np.sin(2 * np.pi * (200 + 1800 * t) * t) + 0.02 * rng.standard_normal(sc.N_SAMPLES)
    y[9000:12000] *= 0.01
    y[12000:13000] = 0.0
    y[14000:14500] *= 3.0
    pcm = np.clip(np.round(y * 32768), -32768, 32767).astype(np.int16)
    ref = sc.mfcc(sc.to_float(pcm[None]))[0].numpy()

    # mfcc_to_s8 on the reference q16 values, so it is tested on its own
    mean, std, in_scale, zp = rng.standard_normal(10) * 50, rng.uniform(5, 50, 10), 0.03, -4
    m0, shift = np.array([quant.quantize_multiplier(1.0 / (s * in_scale * 65536.0)) for s in std]).T
    q16 = quant.round_away(ref.astype(np.float64) * 65536).astype(np.int64).ravel()
    mean_q16 = quant.round_away(mean * 65536).astype(np.int64)
    k = np.arange(q16.size) % 10
    y8 = np.clip(quant.requant(q16 - mean_q16[k], m0[k], shift[k]) + zp, -128, 127)
    return [
        c_array("int16_t", "mfcc_pcm", pcm), c_array("float", "mfcc_ref", ref),
        c_array("int32_t", "m8_in", q16), c_array("int32_t", "m8_mean", mean_q16),
        c_array("int32_t", "m8_mult", m0), c_array("int8_t", "m8_shift", shift),
        f"#define M8_ZP ({zp})\n", c_array("int8_t", "m8_y", y8),
    ]


def main():
    rng = np.random.default_rng(1234)
    body = ['#include "quantize.h"\n\n'] + fp32_vectors(rng) + int8_vectors(rng) + mfcc_vectors(rng)
    OUT.write_text(c_header("TEST_VECTORS", "".join(body)))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
