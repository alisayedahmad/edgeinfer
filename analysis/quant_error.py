"""where int8 quantization error appears, layer by layer.

    python -m analysis.quant_error --samples 64

runs the same inputs through fp32 and int8 and compares every layer output,
not just the logits. the c engine dumps its taps directly (edgeinfer taps),
so its numbers are exact. tflite exposes intermediates through
experimental_preserve_all_tensors and is matched to the fp32 model by
execution order. error is reported relative to the layer's own fp32 range,
since absolute volts mean nothing across layers of different scale.
"""
import argparse
import json
import subprocess

import numpy as np

from analysis import style
from runtime import profile


def c_taps(mode, x, samples):
    """(layer names, taps) for the c engine, taps is (samples, layer, values)."""
    cli, out = str(profile.c_cli()), profile.ARTIFACTS / f"taps_{mode}.f32"
    features = str(profile.ARTIFACTS / "test_features.f32")
    subprocess.run([cli, "taps", mode, features, str(samples), str(out)], check=True, capture_output=True)
    plan = json.loads(subprocess.run([cli, "plan", mode], capture_output=True, text=True, check=True).stdout)
    sizes, names = [], []
    for tensor in plan["tensors"][1:]:
        names.append(tensor["name"])
        sizes.append(tensor["size"] // (1 if mode == "int8" else 4))
    flat = np.fromfile(out, np.float32).reshape(samples, -1)
    taps, offset = [], 0
    for size in sizes:
        taps.append(flat[:, offset:offset + size])
        offset += size
    return names, taps


def c_engine_error(samples):
    x, _ = profile.eval_set()
    names, fp32 = c_taps("fp32", x, samples)
    _, int8 = c_taps("int8", x, samples)
    rows = []
    for name, a, b in zip(names, fp32, int8):
        spread = float(a.max() - a.min()) or 1.0
        rows.append({"layer": name, "max_abs": float(np.abs(a - b).max()),
                     "rel_percent": float(np.abs(a - b).max() / spread * 100),
                     "rms_percent": float(np.sqrt(((a - b) ** 2).mean()) / spread * 100)})
    return rows


# these pad or move data instead of computing, so they are not layers, and
# dropping them lines tflite's sequence up with the c engine's one for one
LAYOUT_OPS = {"PAD", "RESHAPE", "TRANSPOSE", "QUANTIZE", "DEQUANTIZE", "SHAPE", "CAST"}


def dequantize(tensor, params):
    """back to float, per-axis included, since a tflite tensor can carry a scale per channel."""
    tensor = tensor.astype(np.float32)
    scales = np.asarray(params["scales"])
    if not scales.size:
        return tensor
    shape = [1] * tensor.ndim
    shape[params["quantized_dimension"]] = scales.size
    return (tensor - np.asarray(params["zero_points"]).reshape(shape)) * scales.reshape(shape)


def tflite_outputs(path, x, samples):
    """every operator output of one model, keyed by tensor name."""
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(path), experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    details = {d["index"]: d for d in interp.get_tensor_details()}
    # the tensor list also holds the weights, and only the op table says which
    # tensor an operator writes, so this is the one place a private call earns itself
    produced = [i for op in interp._get_ops_details() if op["op_name"] not in LAYOUT_OPS
                for i in op["outputs"]]
    inp = interp.get_input_details()[0]
    values = {}
    for sample in x[:samples]:
        v = sample.reshape(inp["shape"])
        if inp["dtype"] == np.int8:
            scale, zp = inp["quantization"]
            v = np.clip(np.round(v / scale) + zp, -128, 127)
        interp.set_tensor(inp["index"], v.astype(inp["dtype"]))
        interp.invoke()
        for index in produced:
            detail = details[index]
            name = detail["name"] or f"tensor {index}"
            values.setdefault(name, []).append(
                dequantize(interp.get_tensor(index), detail["quantization_parameters"]).ravel())
    return {name: np.stack(rows) for name, rows in values.items()}


def tflite_error(samples):
    """same comparison for tflite, matched by tensor name rather than by order."""
    try:
        from runtime import tflite_runner
    except ImportError:
        return []
    paths = {p: profile.ARTIFACTS / "tflite" / tflite_runner.MODELS[p] for p in ("fp32", "int8")}
    if not all(p.exists() for p in paths.values()):
        return []

    x, _ = profile.eval_set()
    fp32, int8 = tflite_outputs(paths["fp32"], x, samples), tflite_outputs(paths["int8"], x, samples)
    rows = []
    for name, a in fp32.items():
        b = int8.get(name)
        if b is None or a.shape != b.shape:
            continue
        spread = float(a.max() - a.min()) or 1.0
        rows.append({"layer": name.removeprefix("wa/"), "max_abs": float(np.abs(a - b).max()),
                     "rel_percent": float(np.abs(a - b).max() / spread * 100),
                     "rms_percent": float(np.sqrt(((a - b) ** 2).mean()) / spread * 100)})
    return rows


def chart(series):
    """heatmap of relative error, layers across, runtimes down."""
    style.setup()
    layers = max((len(rows) for rows in series.values()), default=0)
    fig, ax = style.plt.subplots(figsize=(0.85 * layers + 3.4, 1.0 * len(series) + 1.8))
    worst = max((r["rel_percent"] for rows in series.values() for r in rows), default=1.0)
    for row, (name, rows) in enumerate(series.items()):
        for col, entry in enumerate(rows):
            value = entry["rel_percent"]
            ax.add_patch(style.plt.Rectangle((col, row), 0.94, 0.9, facecolor=style.sequential(value / worst)))
            ax.text(col + 0.47, row + 0.45, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if value > 0.55 * worst else style.INK)
    first = next(iter(series.values()))
    ax.set_xticks([i + 0.47 for i in range(len(first))], [r["layer"] for r in first], rotation=45, ha="right")
    ax.set_yticks([i + 0.45 for i in range(len(series))], list(series))
    ax.set_xlim(0, layers)
    ax.set_ylim(len(series), 0)
    # square cells, so one runtime does not stretch into a row of bars
    ax.set_aspect("equal")
    ax.grid(visible=False)
    ax.set_title(f"int8 vs fp32 error per layer, % of that layer's fp32 range (max {worst:.2f}%)")
    return style.save(fig, "quant_error")


def main():
    p = argparse.ArgumentParser(description="per-layer int8 error")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--no-chart", action="store_true")
    args = p.parse_args()

    series = {"c engine": c_engine_error(args.samples)}
    tflite = tflite_error(args.samples)
    if tflite:
        series["tflite"] = tflite

    text = "# int8 error by layer\n\nerror relative to each layer's own fp32 range.\n\n"
    for name, rows in series.items():
        text += f"## {name}\n\n" + style.markdown(
            ["layer", "max abs", "max % of range", "rms % of range"],
            [[r["layer"], f"{r['max_abs']:.4f}", f"{r['rel_percent']:.3f}", f"{r['rms_percent']:.3f}"]
             for r in rows]) + "\n"
    style.write_table("quant_error", text)
    if not args.no_chart:
        chart(series)
    print(text)


if __name__ == "__main__":
    main()
