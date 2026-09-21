"""comparison tables and the operator time-share chart.

    python -m analysis.compare_runtimes

reads every results/profiling/*.json a runner wrote and produces
results/tables/runtimes.md plus results/plots/op_time_share.png. peak ram is
measured differently per runtime, so the table prints where each number came
from instead of pretending they are the same measurement.
"""
import argparse
import json
from collections import defaultdict

from analysis import style
from runtime import profile

ORDER = ["pytorch", "onnxruntime", "tflite", "tensorrt", "c_engine"]


def sort_key(record):
    runtime = record["runtime"]
    return (ORDER.index(runtime) if runtime in ORDER else len(ORDER), record["precision"])


def op_share(record):
    # ms per operator category, in the fixed order
    share = defaultdict(float)
    for op in record["ops"]:
        share[style.op_bucket(op["op"])] += op["ms"]
    return share


def summary_table(records):
    rows = []
    for r in records:
        ram = f"{r['peak_ram_kb']:.0f} ({r['peak_ram_source'].split()[0]})" if r.get("peak_ram_kb") else "n/a"
        rows.append([
            r["runtime"], r["precision"], f"{r.get('accuracy', float('nan')) * 100:.2f}",
            f"{r['latency_ms']['p50']:.3f}", f"{r['latency_ms']['p90']:.3f}",
            f"{r['model_size_kb']:.1f}", ram, len(r.get("fused") or []),
        ])
    return style.markdown(
        ["runtime", "precision", "accuracy %", "p50 ms", "p90 ms", "size kb", "peak ram kb", "kernels"], rows)


def operator_table(records):
    rows = []
    for r in records:
        if not r["ops"]:
            continue
        share = op_share(r)
        total = sum(share.values()) or 1.0
        rows.append([f"{r['runtime']} {r['precision']}"] +
                    [f"{share[op]:.3f} ({share[op] / total * 100:4.1f}%)" if share[op] else "-"
                     for op in style.OPS])
    return style.markdown(["runtime", *style.OPS], rows)


def embedded_table(path):
    if not path.exists():
        return ""
    data = json.loads(path.read_text())
    rows = [["flash", f"{data['flash_bytes'] / 1024:.1f} kb", f"{data['flash_limit'] // 1024} kb",
             "fits" if data["flash_fits"] else "OVER"],
            ["ram (bss + stack)", f"{data['ram_bytes'] / 1024:.1f} kb", f"{data['ram_limit'] // 1024} kb",
             "fits" if data["ram_fits"] else "OVER"]]
    for mode, plans in data["arena_bytes"].items():
        rows.append([f"arena {mode}", f"{plans['greedy'] / 1024:.1f} kb greedy",
                     f"{plans['naive'] / 1024:.1f} kb naive", ""])
    table = style.markdown(["cortex-m4", "used", "budget", ""], rows)
    runs = data.get("qemu", [])
    if runs:
        table += "\n" + style.markdown(
            ["mode", "mfcc (m instructions)", "inference (m instructions)", "predicted"],
            [[r["mode"], f"{r['mfcc_instructions'] / 1e6:.2f}", f"{r['infer_instructions'] / 1e6:.2f}",
              f"{r['pred']} (label {r['label']})"] for r in runs])
    return table


def chart(records):
    """where each runtime spends its time: absolute ms, then share of total.

    the runtimes differ by more than an order of magnitude, so one shared
    linear axis would flatten the fast ones into invisible slivers. the share
    panel keeps their breakdown readable next to the magnitude panel.
    """
    # a runtime with no per-op profiler would draw an empty bar that reads as zero
    records = [r for r in records if r["ops"]]
    style.setup()
    fig, (left, right) = style.plt.subplots(1, 2, figsize=(1.5 * len(records) + 5.0, 4.4))
    labels = [f"{r['runtime']}\n{r['precision']}" for r in records]
    totals = [sum(op_share(r).values()) or 1.0 for r in records]
    bottoms, shares = [0.0] * len(records), [0.0] * len(records)
    for op in style.OPS:
        values = [op_share(r)[op] for r in records]
        if not any(values):
            continue
        percent = [v / t * 100 for v, t in zip(values, totals)]
        # a surface-coloured edge keeps a gap between stacked segments
        for ax, vals, base in ((left, values, bottoms), (right, percent, shares)):
            ax.bar(labels, vals, 0.62, bottom=base, label=op, color=style.OP_COLOR[op],
                   edgecolor=style.SURFACE, linewidth=1.5)
        for i, (value, base) in enumerate(zip(percent, shares)):
            if value >= 8:
                right.text(i, base + value / 2, f"{value:.0f}%", ha="center", va="center",
                           fontsize=8, color=style.INK)
        bottoms = [b + v for b, v in zip(bottoms, values)]
        shares = [b + v for b, v in zip(shares, percent)]
    left.set_ylabel("inference time per sample (ms)")
    left.set_title("cost per operator")
    right.set_ylabel("share of inference time (%)")
    right.set_title("where the time goes")
    for ax in (left, right):
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8)
    right.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="operator")
    return style.save(fig, "op_time_share")


def main():
    p = argparse.ArgumentParser(description="runtime comparison tables and chart")
    p.add_argument("--no-chart", action="store_true")
    args = p.parse_args()

    records = sorted(profile.load_all(), key=sort_key)
    if not records:
        raise SystemExit("no results in results/profiling, run the runners first")
    text = ("# runtime comparison\n\nds-cnn keyword spotter, speech commands v2, batch 1.\n\n"
            + summary_table(records)
            + "\n## time per operator (ms)\n\n" + operator_table(records))
    silent = sorted({r["runtime"] for r in records if not r["ops"]})
    if silent:
        text += f"\nno per-operator breakdown for {', '.join(silent)} on this machine.\n"
    embedded = embedded_table(profile.RESULTS / "embedded.json")
    if embedded:
        text += "\n## cortex-m4 target\n\n" + embedded
    style.write_table("runtimes", text)
    if not args.no_chart:
        chart(records)
    print(text)


if __name__ == "__main__":
    main()
