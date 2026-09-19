"""speech commands v2 download, splits and mfcc features.

the mfcc here is the reference for everything downstream: training, all four
runtimes and the fixed-point version in c_engine/src/mfcc.c.
"""
import argparse
import functools
import hashlib
import io
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
import scipy.fft
import torch

ROOT = Path(__file__).resolve().parent

# download.tensorflow.org serves a cert for *.storage.googleapis.com, so go
# through the bucket path directly to keep tls verification on
URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
MD5 = "6b74f3901214cb2c2934e98196829835"

WORDS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero",
]

SR = 16000
N_SAMPLES = 16000
N_FFT = 512
HOP = 320
N_MELS = 40
N_MFCC = 10
FMIN, FMAX = 20.0, 4000.0
AMIN = 1e-10


def download(root=ROOT):
    out = root / "raw" / URL.rsplit("/", 1)[1]
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".part")
    md5 = hashlib.md5()
    with urllib.request.urlopen(URL) as r, open(part, "wb") as f:
        total, done = int(r.headers["Content-Length"]), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            md5.update(chunk)
            done += len(chunk)
            print(f"\rdownloading {done >> 20}/{total >> 20} MB", end="", flush=True)
    print()
    if md5.hexdigest() != MD5:
        raise RuntimeError(f"md5 mismatch on {part}, delete it and retry")
    part.rename(out)
    return out


def read_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        if (w.getsampwidth(), w.getnchannels(), w.getframerate()) != (2, 1, SR):
            raise ValueError("expected 16-bit mono 16 kHz wav")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def prepare(root=ROOT):
    """unpack the archive into memory-mapped int16 arrays under data/cache.

    one pass over the tarball appends every clip to a scratch file, then each
    split is copied out of it row by row, so only one clip is ever held in
    memory and the whole thing runs in a few hundred megabytes. writes
    {train,val,test}.i16 with the audio zero-padded to 1 s, a matching .npz of
    labels and paths, and noise.npy with the background recordings.
    """
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    scratch = cache / "all.i16"
    names, noise, lists = [], [], {}
    clip = np.zeros(N_SAMPLES, dtype=np.int16)
    with tarfile.open(download(root), "r:gz") as tar, open(scratch, "wb") as out:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.removeprefix("./")
            data = tar.extractfile(member).read()
            word = name.split("/")[0]
            if name in ("validation_list.txt", "testing_list.txt"):
                lists[name.split("_")[0]] = set(data.decode().split())
            elif name.endswith(".wav") and word == "_background_noise_":
                noise.append(read_wav(data))
            elif name.endswith(".wav") and word in WORDS:
                x = read_wav(data)[:N_SAMPLES]
                clip[:len(x)] = x
                clip[len(x):] = 0
                out.write(clip.tobytes())
                names.append(name)

    everything = np.memmap(scratch, dtype=np.int16, mode="r", shape=(len(names), N_SAMPLES))
    splits = {"val": lists["validation"], "test": lists["testing"]}
    splits["train"] = set(names) - splits["val"] - splits["test"]
    for split, wanted in splits.items():
        rows = [i for i, name in enumerate(names) if name in wanted]
        audio = np.memmap(cache / f"{split}.i16", dtype=np.int16, mode="w+", shape=(len(rows), N_SAMPLES))
        for out_row, in_row in enumerate(rows):
            audio[out_row] = everything[in_row]
        audio.flush()
        del audio
        paths = [names[i] for i in rows]
        label = np.array([WORDS.index(p.split("/")[0]) for p in paths], dtype=np.int64)
        np.savez(cache / f"{split}.npz", label=label, path=np.array(paths))
        print(f"{split}: {len(rows)} clips")
    del everything
    try:
        scratch.unlink()
    except OSError:
        print(f"could not remove {scratch}, delete it by hand")
    np.save(cache / "noise.npy", np.concatenate(noise))


def load_split(split, root=ROOT):
    """(audio, label) where audio is memory-mapped, so it need not fit in ram."""
    meta = np.load(root / "cache" / f"{split}.npz")
    label = meta["label"]
    audio = np.memmap(root / "cache" / f"{split}.i16", dtype=np.int16, mode="r",
                      shape=(len(label), N_SAMPLES))
    return audio, label


def hz_to_mel(f):
    # slaney scale: linear below 1 kHz, log above
    f = np.asarray(f, dtype=np.float64)
    mel = 3.0 * f / 200.0
    log = f >= 1000.0
    return np.where(log, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mel)


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    hz = 200.0 * m / 3.0
    return np.where(m >= 15.0, 1000.0 * np.exp(np.log(6.4) / 27.0 * (m - 15.0)), hz)


@functools.cache
def mel_filters():
    """(n_mels, n_fft/2 + 1) slaney-normalized filterbank, same as librosa.filters.mel."""
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    edges = mel_to_hz(np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2))
    ramps = edges[:, None] - freqs[None, :]
    fb = np.zeros((N_MELS, len(freqs)))
    for i in range(N_MELS):
        lower = -ramps[i] / (edges[i + 1] - edges[i])
        upper = ramps[i + 2] / (edges[i + 2] - edges[i + 1])
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))
    return fb * (2.0 / (edges[2:] - edges[:-2]))[:, None]


@functools.cache
def dct_matrix():
    return scipy.fft.dct(np.eye(N_MELS), type=2, norm="ortho", axis=0)[:N_MFCC]


@functools.cache
def hann():
    # periodic, matches scipy.signal.get_window("hann", n_fft)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)


def mfcc(audio):
    """mfcc of 1 s clips, (n, 16000) float in [-1, 1) -> (n, 49, 10).

    same as librosa.feature.mfcc(S=librosa.power_to_db(mel, top_db=None),
    n_mfcc=10) with mel = librosa.feature.melspectrogram(y, sr=16000,
    n_fft=512, hop_length=320, center=False, n_mels=40, fmin=20, fmax=4000),
    transposed to (frames, coeffs). runs on whatever device audio lives on.
    """
    x = torch.as_tensor(audio, dtype=torch.float32)
    dev = x.device
    frames = x.unfold(-1, N_FFT, HOP) * torch.as_tensor(hann(), dtype=torch.float32, device=dev)
    power = torch.view_as_real(torch.fft.rfft(frames)).pow(2).sum(-1)
    mel = power @ torch.as_tensor(mel_filters().T, dtype=torch.float32, device=dev)
    logmel = 10.0 * torch.log10(mel.clamp_min(AMIN))
    return logmel @ torch.as_tensor(dct_matrix().T, dtype=torch.float32, device=dev)


def to_float(audio):
    return torch.as_tensor(audio, dtype=torch.float32) / 32768.0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="download and split speech commands v2")
    p.add_argument("--root", type=Path, default=ROOT)
    prepare(p.parse_args().root)
