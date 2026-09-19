"""check the cortex-m4 build against the 1 mb flash / 256 kb ram budget.

    python embedded/constraints.py [--qemu]

section sizes come from arm-none-eabi-size, arena sizes from the host
planner (naive vs greedy), and --qemu also runs the elf and keeps the
per-operator instruction counts. writes results/profiling/embedded.json.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ELF = REPO / "c_engine" / "build" / "arm" / "edgeinfer.elf"
CLI = REPO / "c_engine" / "build" / "host" / "edgeinfer"
OUT = REPO / "results" / "profiling" / "embedded.json"

FLASH_LIMIT = 1024 * 1024
RAM_LIMIT = 256 * 1024
STACK_BYTES = 8 * 1024
# 25 mhz systick under -icount shift=0: one tick per 40 instructions
TICK_INSTRUCTIONS = 40


def sections(elf):
    out = subprocess.run(["arm-none-eabi-size", "-A", str(elf)], capture_output=True, text=True, check=True)
    matches = (re.match(r"(\.\S+)\s+(\d+)\s+\d+", line) for line in out.stdout.splitlines())
    return {m[1]: int(m[2]) for m in matches if m}


def arenas():
    # what each mode needs with and without buffer reuse
    out = {}
    for mode in ["int8", "fp32", "fp32-unfused"]:
        out[mode] = {p: json.loads(subprocess.run([str(CLI), "plan", mode, p], capture_output=True, text=True,
                                                  check=True).stdout)["arena_bytes"]
                     for p in ["greedy", "naive"]}
    return out


def run_qemu():
    # qemu writes semihosting output to stderr
    out = subprocess.run(["sh", str(REPO / "embedded" / "qemu_run.sh"), str(ELF)], capture_output=True, text=True,
                         cwd=REPO, timeout=1800)
    runs = []
    for line in (out.stdout + out.stderr).splitlines():
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        r["mfcc_instructions"] = r.pop("mfcc_ticks") * TICK_INSTRUCTIONS
        r["infer_instructions"] = r.pop("infer_ticks") * TICK_INSTRUCTIONS
        r["steps"] = [{"name": n, "op": op, "instructions": t * TICK_INSTRUCTIONS} for n, op, t in r["steps"]]
        runs.append(r)
    if not runs:
        raise RuntimeError(f"no json from qemu:\n{(out.stdout + out.stderr)[-2000:]}")
    return runs


def main():
    p = argparse.ArgumentParser(description="check flash and ram fit for the cortex-m4 build")
    p.add_argument("--qemu", action="store_true", help="also run the elf and keep per-op instruction counts")
    args = p.parse_args()
    if not ELF.exists():
        raise SystemExit(f"{ELF} missing, run make -C c_engine arm")

    sec = sections(ELF)
    flash = sec.get(".isr_vector", 0) + sec.get(".text", 0) + sec.get(".data", 0)
    ram = sec.get(".data", 0) + sec.get(".bss", 0) + STACK_BYTES
    report = {
        "sections": sec, "flash_bytes": flash, "flash_limit": FLASH_LIMIT,
        "ram_bytes": ram, "ram_limit": RAM_LIMIT, "stack_bytes": STACK_BYTES,
        "flash_fits": flash <= FLASH_LIMIT, "ram_fits": ram <= RAM_LIMIT,
        "arena_bytes": arenas(),
    }
    if args.qemu:
        report["qemu"] = run_qemu()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"flash {flash / 1024:8.1f} kb / {FLASH_LIMIT // 1024} kb  {'fits' if report['flash_fits'] else 'OVER'}")
    print(f"ram   {ram / 1024:8.1f} kb / {RAM_LIMIT // 1024} kb  {'fits' if report['ram_fits'] else 'OVER'}"
          f"  (arena {sec.get('.bss', 0) / 1024:.1f} kb bss incl., stack {STACK_BYTES // 1024} kb)")
    for mode, plans in report["arena_bytes"].items():
        fits = "fits" if plans["greedy"] + STACK_BYTES <= RAM_LIMIT else "OVER"
        print(f"arena {mode:13s} greedy {plans['greedy'] / 1024:8.1f} kb  "
              f"naive {plans['naive'] / 1024:8.1f} kb  {fits}")
    for run in report.get("qemu", []):
        print(f"qemu  {run['mode']:5s} pred {run['pred']} label {run['label']}  "
              f"mfcc {run['mfcc_instructions'] / 1e6:.2f} m instructions  "
              f"inference {run['infer_instructions'] / 1e6:.2f} m instructions")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
