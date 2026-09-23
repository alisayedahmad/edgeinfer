"""the c engine as another runtime: accuracy, latency and per-op timings.

    python -m runtime.c_runner --precision fp32|fp32-unfused|int8

everything comes from c_engine/build/host/edgeinfer, which reads the same
evaluation features as the other runners. latency is measured inside the c
program, so python's call overhead stays out of it. model size is the weight
bytes the mode would ship, and peak ram is the planner's arena, both exact.
"""
import argparse
import json
import subprocess

import numpy as np

from runtime import profile

FEATURES = profile.ARTIFACTS / "test_features.f32"


def cli(*args):
    out = subprocess.run([str(profile.c_cli()), *map(str, args)], capture_output=True, text=True, check=True)
    return out.stdout


def run(precision, runs):
    x, y = profile.eval_set()
    logits_path = profile.ARTIFACTS / f"c_{precision}.f32"
    cli("eval", precision, FEATURES, len(x), logits_path)
    logits = np.fromfile(logits_path, np.float32).reshape(len(x), -1)

    # the same fastest-of-rounds rule the python runners use, see profile.fastest
    bench = profile.fastest(lambda: json.loads(cli("bench", precision, runs)),
                            lambda b: b["latency_ns"]["p50"])
    plan = json.loads(cli("plan", precision))
    layers = [t["name"] for t in plan["tensors"] if t["name"] not in ("input", "pool", "fc")]
    return {
        "runtime": "c_engine", "precision": precision,
        "accuracy": profile.accuracy(logits, y),
        "latency_ms": {k: v / 1e6 for k, v in bench["latency_ns"].items()},
        "model_size_kb": plan["weight_bytes"] / 1024,
        "peak_ram_kb": plan["arena_bytes"] / 1024, "peak_ram_source": "planner arena",
        "ops": [{"name": s["name"], "op": s["op"], "ms": s["ns"] / 1e6, "bytes": s["out_bytes"]}
                for s in bench["steps"]],
        # the fused modes run conv, bn and relu as one kernel per layer
        "fused": [] if precision == "fp32-unfused" else [f"{n} + {n}_bn + {n}_relu" for n in layers],
        "arena_naive_kb": json.loads(cli("plan", precision, "naive"))["arena_bytes"] / 1024,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="c engine inference and profiling")
    p.add_argument("--precision", default="int8", choices=["fp32", "fp32-unfused", "int8"])
    # about two seconds per round, the same target the python runners use
    p.add_argument("--runs", type=int, default=100)
    args = p.parse_args()
    profile.save(run(args.precision, args.runs))
