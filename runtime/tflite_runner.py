"""tflite inference with per-operator profiling.

    python -m runtime.tflite_runner --precision fp32|int8

the python interpreter exposes no profiler, so per-op timings come from
tflite's own benchmark_model with op profiling on (make tools fetches it).
without that binary everything else still works, just with no breakdown.
the int8 model is full integer, so input and output are int8 here.
"""
import argparse
import re
import subprocess

import numpy as np

from runtime import profile

MODELS = {"fp32": "ds_cnn_fp32.tflite", "int8": "ds_cnn_int8.tflite"}
BENCHMARK = profile.REPO / "tools" / "benchmark_model"


def interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter

    interp = Interpreter(model_path=str(path), num_threads=1)
    interp.allocate_tensors()
    return interp


def predict(interp, x):
    """logits for (n, 1, 49, 10) features, quantizing when the model is int8."""
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    scale, zp = inp["quantization"]
    logits = np.empty((len(x), out["shape"][-1]), dtype=np.float32)
    for i, sample in enumerate(x):
        v = sample.reshape(inp["shape"])
        if inp["dtype"] == np.int8:
            v = np.clip(np.round(v / scale) + zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], v.astype(inp["dtype"]))
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.float32)
        if out["dtype"] == np.int8:
            y = (y - out["quantization"][1]) * out["quantization"][0]
        logits[i] = y
    return logits


def benchmark(path, runs):
    """per-op rows and peak memory from benchmark_model, empty if it is missing."""
    if not BENCHMARK.exists():
        print(f"note: {BENCHMARK} missing, skipping the per-op breakdown (make tools)")
        return [], None
    args = [str(BENCHMARK), f"--graph={path}", "--num_threads=1", "--enable_op_profiling=true",
            f"--num_runs={runs}", "--warmup_runs=20"]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout

    ops, columns = [], None
    for line in out.splitlines():
        cells = [c.strip() for c in line.split("\t") if c.strip()]
        if not cells:
            continue
        if "[node type]" in line:
            columns = cells
            continue
        if columns and len(cells) == len(columns) and cells[0].isupper():
            row = dict(zip(columns, cells))
            name = row.get("[Name]", "")
            ops.append({"name": name, "op": profile.canonical_op(name, row["[node type]"]),
                        "ms": float(row["[avg ms]"]), "bytes": int(float(row.get("[mem KB]", 0)) * 1024)})
        if columns and cells[0] == "[Name]":
            break
    peak = re.search(r"Peak memory footprint \(MB\):.*overall=([\d.]+)", out)
    return ops, float(peak[1]) * 1024 if peak else None


def run(precision, runs):
    path = profile.ARTIFACTS / "tflite" / MODELS[precision]
    x, y = profile.eval_set()
    before = profile.rss_kb()
    interp = interpreter(path)
    logits = predict(interp, x)
    rss = profile.rss_kb() - before

    inp = interp.get_input_details()[0]
    sample = x[:1].reshape(inp["shape"])
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        sample = np.clip(np.round(sample / scale) + zp, -128, 127)
    sample = sample.astype(inp["dtype"])

    def once():
        interp.set_tensor(inp["index"], sample)
        interp.invoke()

    ops, peak = benchmark(path, runs)
    return {
        "runtime": "tflite", "precision": precision,
        "accuracy": profile.accuracy(logits, y),
        "latency_ms": profile.latency(once, runs),
        "model_size_kb": path.stat().st_size / 1024,
        "peak_ram_kb": peak if peak else rss,
        "peak_ram_source": "benchmark_model peak footprint" if peak else "process rss high-water delta",
        "ops": ops,
        "fused": [op["name"] for op in ops],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="tflite inference and profiling")
    p.add_argument("--precision", default="fp32", choices=list(MODELS))
    p.add_argument("--runs", type=int, default=200)
    args = p.parse_args()
    profile.save(run(args.precision, args.runs))
