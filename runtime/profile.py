"""shared profiling interface, one json record per runtime and precision.

every runner writes results/profiling/<runtime>_<precision>.json with the
same shape, so analysis/ never has to know which runtime produced what:

    {"runtime": "onnxruntime", "precision": "int8", "accuracy": 0.94,
     "latency_ms": {"p50": .., "p90": .., "p99": .., "mean": ..},
     "model_size_kb": .., "peak_ram_kb": .., "peak_ram_source": "..",
     "ops": [{"name": "conv1", "op": "conv2d", "ms": .., "bytes": ..}],
     "fused": ["conv1 + conv1_bn + conv1_relu", ..],
     "provenance": {"cpu": .., "os": .., "commit": .., "packages": {..}}}

peak ram is not the same measurement everywhere, so peak_ram_source says
where the number came from and the comparison table prints it.
"""
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
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

# the packages whose version can move a latency, checked once per record
TRACKED = ("torch", "numpy", "onnx", "onnxruntime", "onnx2tf", "ai-edge-litert", "tensorrt")


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


def latency(run, runs=200, warmup=20, rounds=3, budget=30.0):
    """wall-clock percentiles of a single-sample inference, in ms.

    reported from the fastest round, not from one pass. this laptop throttles,
    and measuring the same binary twice a minute apart can differ by 4x, which
    is drift between rounds rather than spread inside one. rounds keep going
    for `budget` seconds so they span more than one throttling episode, and
    the fastest of them is the closest estimate of the kernel's own cost.
    """
    for _ in range(warmup):
        run()
    best, done, deadline = None, 0, time.perf_counter() + budget
    while done < rounds or time.perf_counter() < deadline:
        done += 1
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            run()
            times.append((time.perf_counter() - start) * 1e3)
        times = np.sort(times)
        stats = {"p50": float(times[len(times) // 2]), "p90": float(times[int(len(times) * 0.9)]),
                 "p99": float(times[int(len(times) * 0.99)]), "mean": float(times.mean())}
        if best is None or stats["p50"] < best["p50"]:
            best = stats
    return best


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


def git(*args):
    try:
        done = subprocess.run(("git", "-C", str(REPO), *args), capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def cpu_name():
    """the model name, wherever the platform keeps it"""
    system = platform.system()
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def provenance():
    """the machine, the versions and the commit that produced a record.

    a latency on its own is unreadable a few months later, and the rows of
    the comparison table are only comparable while they share this block.
    results/ is excluded from the dirty check because a benchmark writes there.
    """
    versions = {}
    for name in TRACKED:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    commit = git("rev-parse", "--short", "HEAD")
    if commit and git("status", "--porcelain", "--", ":!results"):
        commit += "-dirty"
    block = {
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "cpu": cpu_name(),
        "cores": os.cpu_count(),
        "os": f"{platform.system().lower()} {platform.release()}",
        "python": platform.python_version(),
        "packages": versions,
    }
    marker = ARTIFACTS / "smoke"
    if marker.exists():
        block["smoke"] = marker.read_text().strip()
    return block


def save(record):
    RESULTS.mkdir(parents=True, exist_ok=True)
    record.setdefault("provenance", provenance())
    path = RESULTS / f"{record['runtime']}_{record['precision']}.json"
    path.write_text(json.dumps(record, indent=2))
    lat = record["latency_ms"]
    print(f"{record['runtime']:12s} {record['precision']:6s} acc {record.get('accuracy', float('nan')):.4f}  "
          f"p50 {lat['p50']:.3f} ms  size {record['model_size_kb']:.1f} kb  -> {path.name}")
    return path


def load_all():
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json")) if p.stem != "embedded"]
