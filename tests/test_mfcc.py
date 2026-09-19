"""mfcc checks: the torch path must equal librosa, and the c path must track it."""
import subprocess

import numpy as np
import pytest

from data import speech_commands as sc
from runtime import profile
from train.ds_cnn import N_FRAMES


def clips(n=8, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(sc.N_SAMPLES) / sc.SR
    out = []
    for i in range(n):
        y = 0.5 * np.sin(2 * np.pi * (120 + 130 * i + 1400 * t) * t) + 0.01 * rng.standard_normal(sc.N_SAMPLES)
        # a quiet stretch and digital silence, where fixed point hurts most
        y[3000:5000] *= 0.01
        y[9000:9500] = 0.0
        out.append(np.clip(np.round(y * 32768), -32768, 32767))
    return np.array(out, dtype=np.int16)


def test_matches_librosa():
    librosa = pytest.importorskip("librosa")
    audio = clips(4)
    ours = sc.mfcc(sc.to_float(audio)).numpy()
    for i, pcm in enumerate(audio):
        mel = librosa.feature.melspectrogram(
            y=pcm.astype(np.float32) / 32768.0, sr=sc.SR, n_fft=sc.N_FFT, hop_length=sc.HOP,
            center=False, n_mels=sc.N_MELS, fmin=sc.FMIN, fmax=sc.FMAX)
        ref = librosa.feature.mfcc(S=librosa.power_to_db(mel, top_db=None), n_mfcc=sc.N_MFCC).T
        assert np.abs(ours[i] - ref).max() < 1e-2


def test_filterbank_matches_librosa():
    librosa = pytest.importorskip("librosa")
    ref = librosa.filters.mel(sr=sc.SR, n_fft=sc.N_FFT, n_mels=sc.N_MELS, fmin=sc.FMIN, fmax=sc.FMAX)
    assert np.abs(ref - sc.mel_filters()).max() < 1e-6


def test_shape_and_silence():
    out = sc.mfcc(sc.to_float(np.zeros((2, sc.N_SAMPLES), np.int16))).numpy()
    assert out.shape == (2, N_FRAMES, sc.N_MFCC)
    # digital silence floors at 10 * log10(amin) in every mel band
    floor = 10 * np.log10(sc.AMIN) * np.sqrt(sc.N_MELS)
    assert np.allclose(out[:, :, 0], floor, atol=1e-2)
    assert np.allclose(out[:, :, 1:], 0.0, atol=1e-2)


def test_fixed_point_tracks_the_reference(tmp_path):
    """c_engine/src/mfcc.c is integer only, so it drifts, but not far."""
    try:
        cli = profile.c_cli()
    except SystemExit:
        pytest.skip("c engine cli not built")
    audio = clips(8)
    pcm, out = tmp_path / "clips.s16", tmp_path / "clips.f32"
    audio.astype("<i2").tofile(pcm)
    subprocess.run([str(cli), "mfcc", str(pcm), str(len(audio)), str(out)], check=True, capture_output=True)
    got = np.fromfile(out, np.float32).reshape(len(audio), N_FRAMES, sc.N_MFCC)
    err = np.abs(got - sc.mfcc(sc.to_float(audio)).numpy())
    assert err.max() < 0.5, f"max {err.max():.3f} db"
    assert err.mean() < 0.05, f"mean {err.mean():.4f} db"
