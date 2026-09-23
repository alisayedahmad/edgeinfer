"""onnx runtime inference with per-operator profiling.

    python -m runtime.onnx_runner --precision fp32|int8|fp32-unopt

per-op timing and memory come from onnxruntime's own trace, where every node
event carries its duration and the bytes it allocated. the optimized graph is
written next to the model so fusion_analysis.py can see what ort merged.
fp32-unopt disables graph optimizations, which is the unfused baseline.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from runtime import profile

MODELS = {"fp32": "ds_cnn.onnx", "fp32-unopt": "ds_cnn.onnx", "int8": "ds_cnn_int8.onnx"}
LEVELS = {"fp32": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
          "int8": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
          "fp32-unopt": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}


def session(path, level, profile_dir=None, optimized_path=None):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = level
    # single threaded, so per-op times add up to the end-to-end latency
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    if profile_dir:
        opts.enable_profiling = True
        opts.profile_file_prefix = str(Path(profile_dir) / "ort")
    if optimized_path:
        opts.optimized_model_filepath = str(optimized_path)
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def profile_ops(path, level, x, runs=50):
    """per-node ms and output bytes, summed over runs of one sample."""
    with tempfile.TemporaryDirectory() as tmp:
        sess = session(path, level, profile_dir=tmp)
        name = sess.get_inputs()[0].name
        for _ in range(runs):
            sess.run(None, {name: x[:1]})
        trace = json.loads(Path(sess.end_profiling()).read_text())
    ops = {}
    for event in trace:
        if event.get("cat") != "Node" or not event["name"].endswith("_kernel_time"):
            continue
        node, args = event["name"].removesuffix("_kernel_time"), event.get("args", {})
        rec = ops.setdefault(node, {"name": node, "op": profile.canonical_op(node, args.get("op_name")),
                                    "ms": 0.0, "bytes": int(args.get("output_size", 0))})
        rec["ms"] += event["dur"] / 1e3 / runs
    return list(ops.values())


def run(precision, runs, batch):
    path = profile.ARTIFACTS / MODELS[precision]
    x, y = profile.eval_set()
    # the eval set is loaded first on purpose: the figure has to cover the
    # runtime and its arena, not the batch of inputs a runner happens to use
    before = profile.rss_kb()
    optimized = profile.ARTIFACTS / f"ds_cnn_ort_{precision}.onnx"
    sess = session(path, LEVELS[precision], optimized_path=optimized)
    name = sess.get_inputs()[0].name
    sess.run(None, {name: x[:1]})
    peak = profile.rss_kb() - before

    logits = np.concatenate([sess.run(None, {name: x[i:i + batch]})[0] for i in range(0, len(x), batch)])

    graph = onnx.load(optimized).graph
    return {
        "runtime": "onnxruntime", "precision": precision,
        "accuracy": profile.accuracy(logits, y),
        "latency_ms": profile.latency(lambda: sess.run(None, {name: x[:1]}), runs),
        "model_size_kb": path.stat().st_size / 1024,
        "peak_ram_kb": peak, "peak_ram_source": "rss for the runtime plus one inference",
        "ops": profile_ops(path, LEVELS[precision], x),
        "fused": [n.name for n in graph.node],
        "nodes_before": len(onnx.load(path).graph.node), "nodes_after": len(graph.node),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="onnx runtime inference and profiling")
    p.add_argument("--precision", default="fp32", choices=list(MODELS))
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    profile.save(run(args.precision, args.runs, args.batch))
