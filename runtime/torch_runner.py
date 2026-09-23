"""the pytorch model as another runtime, the reference every other row answers to.

    python -m runtime.torch_runner

no per-operator breakdown on purpose: torch's profiler reports its module
tree, not the kernels a runtime schedules, so the numbers would not line up
with the other runtimes. model size is the parameter bytes, which is what an
export carries, and the eval set is the same one every runner reads.
"""
import argparse

import torch

from runtime import profile
from train.ds_cnn import load


def run(runs, batch):
    # single threaded, like every other runtime here
    torch.set_num_threads(1)
    model, _ = load(profile.ARTIFACTS / "ds_cnn.pt")
    x, y = profile.eval_set()

    before = profile.rss_kb()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            logits.append(model(torch.from_numpy(x[start:start + batch])).numpy())
    peak = profile.rss_kb() - before
    logits = [row for chunk in logits for row in chunk]

    sample = torch.from_numpy(x[:1])

    def once():
        with torch.inference_mode():
            model(sample)

    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "runtime": "pytorch", "precision": "fp32",
        "accuracy": profile.accuracy(logits, y),
        "latency_ms": profile.latency(once, runs),
        "model_size_kb": weights / 1024,
        "peak_ram_kb": peak, "peak_ram_source": "process rss high-water delta",
        "ops": [], "fused": [],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="pytorch reference inference")
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()
    profile.save(run(args.runs, args.batch))
