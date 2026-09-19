"""shared profiling interface, one json record per runtime and precision.

every runner writes results/profiling/<runtime>_<precision>.json with the
same shape, so analysis/ never has to know which runtime produced what:

    {"runtime": "onnxruntime", "precision": "int8", "accuracy": 0.94,
     "latency_ms": {"p50": .., "p90": .., "p99": .., "mean": ..},
     "model_size_kb": .., "peak_ram_kb": .., "peak_ram_source": "..",
     "ops": [{"name": "conv1", "op": "conv2d", "ms": .., "bytes": ..}],
     "fused": ["conv1 + conv1_bn + conv1_relu", ..]}

peak ram is not the same measurement everywhere, so peak_ram_source says
where the number came from and the comparison table prints it.
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
RESULTS = REPO / "results" / "profiling"

# our onnx node names carry the layer, which is the most reliable signal
LAYER_OP = {"conv1": "conv2d", "dw": "depthwise_conv", "pw": "pointwise_conv", "pool": "pool", "fc": "dense"}
TYPE_OP = {
    "conv": "conv2d", "conv2d": "conv2d", "qlinearconv": "conv2d", "convolution": "conv2d",
    "depthwise_conv_2d": "depthwise_conv", "depthwiseconv2dnative": "depthwise_conv",
    "batchnormalization": "batch_norm", "fusedbatchnormv3": "batch_norm",
    "relu": "relu", "clip": "relu",
    "globalaveragepool": "pool", "average_pool_2d": "pool", "mean": "pool", "reducemean": "pool",
    "gemm": "dense", "matmul": "dense", "fully_connected": "dense",
    "quantizelinear": "quantize", "dequantizelinear": "quantize", "quantize": "quantize", "dequantize": "quantize",
    "reshape": "other", "flatten": "other", "transpose": "other", "cast": "other", "shape": "other",
}


def canonical_op(name, op_type=""):
    """map a runtime's node name and op type onto our operator categories.

    the op type decides whenever it is unambiguous. convolutions are not: only
    the layer name says whether a conv is the first layer, a depthwise or a
    pointwise, and that name survives into onnx, tflite and tensorrt.
    """
    base = TYPE_OP.get((op_type or "").lower().removeprefix("tfl."))
    if base not in (None, "conv2d"):
        return base
    # the lookaround keeps "fc" out of "mfcc" and matches ort's dw1_token_4
    hit = re.search(r"(?<![A-Za-z0-9_])(conv1|dw\d+|pw\d+|pool|fc)(?![0-9])", name or "")
    if hit:
        return LAYER_OP[hit[1] if hit[1] in LAYER_OP else re.sub(r"\d+$", "", hit[1])]
    return base or "other"


def c_cli():
    # gcc appends .exe on windows, and both builds can share one tree
    for name in ("edgeinfer.exe", "edgeinfer") if os.name == "nt" else ("edgeinfer", "edgeinfer.exe"):
        path = REPO / "c_engine" / "build" / "host" / name
        if path.exists():
            return path
    raise SystemExit("c engine cli missing, run make -C c_engine")


def eval_set():
    d = np.load(ARTIFACTS / "test_features.npz")
    return d["x"], d["y"]


def calib_set():
    return np.load(ARTIFACTS / "calib_features.npy")


def latency(run, runs=200, warmup=20):
    """wall-clock percentiles of a single-sample inference, in ms."""
    for _ in range(warmup):
        run()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1e3)
    times = np.sort(times)
    return {"p50": float(times[len(times) // 2]), "p90": float(times[int(len(times) * 0.9)]),
            "p99": float(times[int(len(times) * 0.99)]), "mean": float(times.mean())}


def accuracy(logits, y):
    return float((np.asarray(logits).argmax(-1) == np.asarray(y)).mean())


def rss_kb():
    # high-water mark of this process in kb, 0 where the platform has no api
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        pass
    try:
        import psutil

        return psutil.Process().memory_info().peak_wset // 1024
    except ImportError:
        return 0


def save(record):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{record['runtime']}_{record['precision']}.json"
    path.write_text(json.dumps(record, indent=2))
    lat = record["latency_ms"]
    print(f"{record['runtime']:12s} {record['precision']:6s} acc {record.get('accuracy', float('nan')):.4f}  "
          f"p50 {lat['p50']:.3f} ms  size {record['model_size_kb']:.1f} kb  -> {path.name}")
    return path


def load_all():
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json")) if p.stem != "embedded"]
