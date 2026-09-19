"""fixed-point mfcc against the librosa reference.

    python -m analysis.mfcc_error --clips 32

runs the same clips through c_engine/src/mfcc.c (integer only, q15 fft) and
through the float reference in data/speech_commands.py, then reports the
error per coefficient and draws it across time frames. the error lives in
quiet frames, where 16-bit fft rounding noise is close to the signal itself.
"""
import argparse
import subprocess

import numpy as np

from analysis import style
from data import speech_commands as sc
from runtime import profile
from train.ds_cnn import N_FRAMES


def clips(n, seed=0):
    """(n, 16000) int16 clips: real recordings when the dataset is there."""
    try:
        audio, _ = sc.load_split("test")
        return audio[np.random.default_rng(seed).choice(len(audio), n, replace=False)]
    except FileNotFoundError:
        rng = np.random.default_rng(seed)
        t = np.arange(sc.N_SAMPLES) / sc.SR
        out = []
        for i in range(n):
            y = 0.5 * np.sin(2 * np.pi * (150 + 90 * i + 1500 * t) * t) + 0.01 * rng.standard_normal(sc.N_SAMPLES)
            y[: 2000 + 200 * i] *= 0.02
            out.append(np.clip(np.round(y * 32768), -32768, 32767))
        return np.array(out, dtype=np.int16)


def c_mfcc(audio):
    pcm = profile.ARTIFACTS / "mfcc_clips.s16"
    out = profile.ARTIFACTS / "mfcc_clips.f32"
    profile.ARTIFACTS.mkdir(exist_ok=True)
    audio.astype("<i2").tofile(pcm)
    subprocess.run([str(profile.c_cli()), "mfcc", str(pcm), str(len(audio)), str(out)],
                   check=True, capture_output=True)
    return np.fromfile(out, np.float32).reshape(len(audio), N_FRAMES, sc.N_MFCC)


def chart(err):
    """error across frames, one line per coefficient is unreadable, so show the
    worst and the mean band instead."""
    style.setup()
    fig, (left, right) = style.plt.subplots(1, 2, figsize=(11.0, 3.8))
    frames = np.arange(err.shape[1])
    left.fill_between(frames, err.mean(axis=(0, 2)), err.max(axis=(0, 2)), color=style.CATEGORICAL[0], alpha=0.25)
    left.plot(frames, err.max(axis=(0, 2)), color=style.CATEGORICAL[0], linewidth=2, label="worst coefficient")
    left.plot(frames, err.mean(axis=(0, 2)), color=style.CATEGORICAL[1], linewidth=2, label="mean")
    left.set_xlabel("frame (20 ms hop)")
    left.set_ylabel("|fixed point - librosa| (db)")
    left.set_title("error across the clip")
    left.legend()

    coeffs = np.arange(err.shape[2])
    worst, mean = err.max(axis=(0, 1)), err.mean(axis=(0, 1))
    right.bar(coeffs - 0.19, worst, 0.34, color=style.CATEGORICAL[0], label="worst")
    right.bar(coeffs + 0.19, mean, 0.34, color=style.CATEGORICAL[2], label="mean")
    for c, v in zip(coeffs, worst):
        right.text(c - 0.19, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=style.INK)
    right.set_xticks(coeffs)
    right.set_xlabel("mfcc coefficient")
    right.set_ylabel("|fixed point - librosa| (db)")
    right.set_title("error per coefficient")
    right.grid(axis="x", visible=False)
    right.legend()
    return style.save(fig, "mfcc_error")


def main():
    p = argparse.ArgumentParser(description="fixed-point mfcc vs the float reference")
    p.add_argument("--clips", type=int, default=32)
    p.add_argument("--no-chart", action="store_true")
    args = p.parse_args()

    audio = clips(args.clips)
    err = np.abs(c_mfcc(audio) - sc.mfcc(sc.to_float(audio)).numpy())
    rows = [[c, f"{err[:, :, c].max():.4f}", f"{err[:, :, c].mean():.4f}"] for c in range(err.shape[2])]
    text = ("# fixed-point mfcc error\n\n"
            f"{len(audio)} clips, 49 frames each, against librosa in db.\n\n"
            f"max {err.max():.4f} db, mean {err.mean():.4f} db\n\n"
            + style.markdown(["coefficient", "max db", "mean db"], rows))
    style.write_table("mfcc_error", text)
    if not args.no_chart:
        chart(err)
    print(text)


if __name__ == "__main__":
    main()
