"""buffer lifetimes and what the memory planner saves.

    python -m analysis.memory_timeline --mode int8

the plan comes from the c engine's own planner (edgeinfer plan), so the
picture is the allocation the engine really runs. each row is an
intermediate tensor, drawn from the step that writes it to the step that
last reads it, coloured by the physical buffer behind it. the naive panel
gives every tensor its own buffer, which is the number to beat.
"""
import argparse
import json
import subprocess

import numpy as np

from analysis import style
from runtime import profile


def plan(mode, planner):
    args = [str(profile.c_cli()), "plan", mode] + (["naive"] if planner == "naive" else [])
    return json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)


def timeline(ax, data, title, color_by_buffer):
    tensors = data["tensors"]
    steps = data["n_steps"]
    for row, tensor in enumerate(tensors):
        # the input exists before step 0, the output lives past the last step
        start = max(tensor["first"], -0.5)
        end = min(tensor["last"], steps) + 0.5
        color = style.CATEGORICAL[tensor["buffer"] % len(style.CATEGORICAL)] if color_by_buffer else style.MUTED
        ax.barh(row, end - start, left=start, height=0.62, color=color,
                edgecolor=style.SURFACE, linewidth=1.2)
        size = tensor["size"]
        label = f"{size} b" if size < 1024 else f"{size / 1024:.0f} kb"
        ax.text(end + 0.15, row, label, va="center", fontsize=7.5, color=style.MUTED)
    ax.set_yticks(range(len(tensors)), [t["name"] for t in tensors], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-1.0, steps + 2.2)
    ax.set_xlabel("inference step")
    ax.set_title(f"{title} - {data['arena_bytes'] / 1024:.1f} kb in {data['n_buffers']} buffer"
                 f"{'' if data['n_buffers'] == 1 else 's'}")
    ax.grid(axis="y", visible=False)


def chart(mode):
    greedy, naive = plan(mode, "greedy"), plan(mode, "naive")
    style.setup()
    height = 0.32 * len(greedy["tensors"]) + 1.8
    fig, (left, right) = style.plt.subplots(1, 2, figsize=(12.5, height), sharey=True)
    timeline(left, naive, "one buffer per tensor", color_by_buffer=False)
    timeline(right, greedy, "greedy reuse", color_by_buffer=True)
    fig.suptitle(f"buffer lifetimes, {mode}", y=1.0)
    return style.save(fig, f"memory_timeline_{mode}")


def table(modes):
    rows = []
    for mode in modes:
        greedy, naive = plan(mode, "greedy"), plan(mode, "naive")
        biggest = max(t["size"] for t in greedy["tensors"])
        rows.append([mode, f"{naive['arena_bytes'] / 1024:.1f}", f"{greedy['arena_bytes'] / 1024:.1f}",
                     f"{naive['arena_bytes'] / greedy['arena_bytes']:.1f}x", greedy["n_buffers"],
                     f"{biggest / 1024:.1f}"])
    text = style.markdown(["mode", "naive kb", "planned kb", "saved", "buffers", "largest tensor kb"], rows)

    # what the other runtimes report, measured differently in each case
    runtimes = [[f"{r['runtime']} {r['precision']}", f"{r['peak_ram_kb']:.0f}", r["peak_ram_source"]]
                for r in profile.load_all() if r.get("peak_ram_kb")]
    if runtimes:
        text += "\n" + style.markdown(["runtime", "peak ram kb", "how it was measured"], runtimes)
    return text


def main():
    p = argparse.ArgumentParser(description="memory plan timeline and totals")
    p.add_argument("--mode", default="int8", choices=["fp32", "fp32-unfused", "int8"])
    p.add_argument("--all-modes", action="store_true", help="also chart the other two modes")
    args = p.parse_args()

    modes = ["fp32", "fp32-unfused", "int8"] if args.all_modes else [args.mode]
    for mode in modes:
        chart(mode)
    text = "# memory\n\narena sizes from the c engine planner.\n\n" + table(["fp32", "fp32-unfused", "int8"])
    style.write_table("memory", text)
    print(text)


if __name__ == "__main__":
    main()
