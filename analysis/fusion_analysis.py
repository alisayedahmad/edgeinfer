"""what each runtime fused, and what the fusion bought.

    python -m analysis.fusion_analysis

three things get compared: the graph torch.onnx produces (its dynamo path
folds batchnorm before any runtime sees it), the graph each runtime actually
runs, and the cost of fusing vs not. latency comes from results/profiling,
the numerical difference from running both paths on the same inputs.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx

from analysis import style
from runtime import profile


def exporter_table():
    # the torchscript export keeps modules as nodes, the dynamo export folds bn
    rows = []
    exports = [("torchscript (ds_cnn.onnx)", "ds_cnn.onnx"), ("dynamo (ds_cnn_dynamo.onnx)", "ds_cnn_dynamo.onnx")]
    for name, path in exports:
        full = profile.ARTIFACTS / path
        if not full.exists():
            continue
        counts = Counter(n.op_type for n in onnx.load(full).graph.node)
        rows.append([name, sum(counts.values()), counts.get("Conv", 0), counts.get("BatchNormalization", 0),
                     counts.get("Relu", 0), ", ".join(f"{k} x{v}" for k, v in counts.most_common() if k not in
                                                      ("Conv", "BatchNormalization", "Relu"))])
    return style.markdown(["export", "nodes", "conv", "batchnorm", "relu", "rest"], rows)


def runtime_table(records):
    rows = []
    for r in records:
        fused = [f for f in (r.get("fused") or []) if " + " in f]
        rows.append([f"{r['runtime']} {r['precision']}", r.get("nodes_before", "-"),
                     r.get("nodes_after") or len(r.get("ops") or []) or "-",
                     len(fused) or "-", (fused or [""])[0][:60]])
    return style.markdown(["runtime", "nodes in", "kernels run", "merged names", "example"], rows)


def onnx_diff(x):
    """max logit difference between ort with and without graph optimizations."""
    from runtime import onnx_runner

    path = profile.ARTIFACTS / "ds_cnn.onnx"
    out = []
    for precision in ["fp32", "fp32-unopt"]:
        sess = onnx_runner.session(path, onnx_runner.LEVELS[precision])
        out.append(sess.run(None, {sess.get_inputs()[0].name: x})[0])
    return float(np.abs(out[0] - out[1]).max())


def impact_table(records, x):
    """the README's fusion table: latency, memory and numerical difference."""
    by_key = {(r["runtime"], r["precision"]): r for r in records}
    rows = []
    pairs = [("c engine, unfused (conv, bn, relu)", ("c_engine", "fp32-unfused")),
             ("c engine, manual fusion", ("c_engine", "fp32")),
             ("onnx runtime, optimizations off", ("onnxruntime", "fp32-unopt")),
             ("onnx runtime, optimizations on", ("onnxruntime", "fp32")),
             ("tensorrt fp16", ("tensorrt", "fp16"))]
    for label, key in pairs:
        r = by_key.get(key)
        if not r:
            continue
        ram = f"{r['peak_ram_kb']:.0f}" if r.get("peak_ram_kb") else "n/a"
        rows.append([label, f"{r['latency_ms']['p50']:.3f}", ram, len(r["ops"])])

    diffs = []
    c_fused, c_unfused = profile.ARTIFACTS / "c_fp32.f32", profile.ARTIFACTS / "c_fp32-unfused.f32"
    if c_fused.exists() and c_unfused.exists():
        a, b = np.fromfile(c_fused, np.float32), np.fromfile(c_unfused, np.float32)
        diffs.append(["c engine fused vs unfused", f"{np.abs(a - b).max():.3e}"])
    if (profile.ARTIFACTS / "ds_cnn.onnx").exists():
        diffs.append(["onnx runtime optimized vs not", f"{onnx_diff(x):.3e}"])
    return (style.markdown(["configuration", "p50 ms", "peak ram kb", "kernels"], rows)
            + "\n" + style.markdown(["numerical difference (max abs logit)", "value"], diffs))


def chart(records):
    """fused vs unfused latency, per runtime that has both."""
    pairs = [("c_engine", "fp32-unfused", "fp32"), ("onnxruntime", "fp32-unopt", "fp32")]
    by_key = {(r["runtime"], r["precision"]): r for r in records}
    labels, unfused, fused = [], [], []
    for runtime, off, on in pairs:
        if (runtime, off) in by_key and (runtime, on) in by_key:
            labels.append(runtime)
            unfused.append(by_key[(runtime, off)]["latency_ms"]["p50"])
            fused.append(by_key[(runtime, on)]["latency_ms"]["p50"])
    if not labels:
        return None

    style.setup()
    fig, ax = style.plt.subplots(figsize=(1.9 * len(labels) + 3.0, 3.8))
    pos = np.arange(len(labels))
    for offset, values, label, color in [(-0.19, unfused, "unfused", style.CATEGORICAL[1]),
                                         (0.19, fused, "fused", style.CATEGORICAL[0])]:
        ax.bar(pos + offset, values, 0.34, label=label, color=color)
        for p, v in zip(pos + offset, values):
            ax.text(p, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=style.INK)
    ax.set_xticks(pos, labels)
    ax.set_ylabel("p50 latency per sample (ms)")
    ax.set_title("conv + bn + relu fused vs run separately")
    ax.grid(axis="x", visible=False)
    ax.legend()
    return style.save(fig, "fusion_impact")


def main():
    p = argparse.ArgumentParser(description="fusion decisions and their cost")
    p.add_argument("--no-chart", action="store_true")
    args = p.parse_args()

    records = profile.load_all()
    if not records:
        raise SystemExit("no results in results/profiling, run the runners first")
    x, _ = profile.eval_set()
    text = ("# fusion\n\n## what the exporter already folded\n\n" + exporter_table()
            + "\n## what each runtime runs\n\n" + runtime_table(records)
            + "\n## fusion impact\n\n" + impact_table(records, x[:64])
            + "\nfusion saves two passes over the activations per layer, so on a desktop cpu with\n"
              "small tensors the win can fall inside run-to-run noise. the deterministic\n"
              "evidence is the operator breakdown in runtimes.md: the batch_norm and relu rows\n"
              "of an unfused run are exactly what fusing removes.\n")
    style.write_table("fusion", text)
    if not args.no_chart:
        chart(records)
    print(text)


if __name__ == "__main__":
    main()
