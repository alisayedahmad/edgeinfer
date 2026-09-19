"""train ds-cnn on speech commands v2, all 35 words.

augmentation happens on raw audio on the training device (time shift and
background noise), then mfcc is computed per batch. writes the best
checkpoint by validation accuracy, then scores it on the test split.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import speech_commands as sc
from train.ds_cnn import DSCNN

REPO = Path(__file__).resolve().parents[1]


def augment(x, noise, shift=1600, noise_prob=0.8, noise_vol=0.1):
    # shift up to 100 ms with zero fill, mix a random 1 s crop of background noise
    n, dev = x.shape[0], x.device
    t = torch.arange(sc.N_SAMPLES, device=dev)
    src = t[None] - torch.randint(-shift, shift + 1, (n, 1), device=dev)
    valid = (src >= 0) & (src < sc.N_SAMPLES)
    x = torch.gather(x, 1, src.clamp(0, sc.N_SAMPLES - 1)) * valid
    start = torch.randint(0, len(noise) - sc.N_SAMPLES, (n, 1), device=dev)
    vol = noise_vol * torch.rand(n, 1, device=dev) * (torch.rand(n, 1, device=dev) < noise_prob)
    return (x + vol * noise[start + t[None]]).clamp(-1.0, 1.0)


def features(audio, mean, std, dev, bs=2048):
    out = [(sc.mfcc(sc.to_float(audio[i:i + bs]).to(dev)) - mean) / std for i in range(0, len(audio), bs)]
    return torch.cat(out).unsqueeze(1)


def feature_stats(audio, dev, bs=2048):
    # per-coefficient mean/std over every frame of the clean training set
    total, sq, n = 0.0, 0.0, 0
    for i in range(0, len(audio), bs):
        f = sc.mfcc(sc.to_float(audio[i:i + bs]).to(dev)).double().flatten(0, 1)
        total, sq, n = total + f.sum(0), sq + (f * f).sum(0), n + len(f)
    mean = total / n
    return mean.float(), (sq / n - mean * mean).sqrt().float()


@torch.no_grad()
def accuracy(model, x, y, bs=1024):
    model.eval()
    pred = torch.cat([model(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)])
    return (pred.cpu() == torch.as_tensor(y)).float().mean().item()


def train(args):
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cudnn.benchmark = True

    audio, label = sc.load_split("train", args.root)
    val_audio, val_label = sc.load_split("val", args.root)
    noise = sc.to_float(np.load(args.root / "cache" / "noise.npy")).to(dev)
    mean, std = feature_stats(audio, dev)
    val_x = features(val_audio, mean, std, dev)

    model = DSCNN(len(sc.WORDS), args.width, args.blocks).to(dev)
    steps_per_epoch = len(audio) // args.bs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    ckpt = {
        "n_classes": len(sc.WORDS), "width": args.width, "blocks": args.blocks,
        "words": sc.WORDS, "feat_mean": mean.cpu().numpy(), "feat_std": std.cpu().numpy(),
    }

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, perm = time.time(), rng.permutation(len(audio))
        for i in range(steps_per_epoch):
            idx = np.sort(perm[i * args.bs:(i + 1) * args.bs])
            x = sc.to_float(audio[idx]).to(dev, non_blocking=True)
            y = torch.from_numpy(label[idx]).to(dev, non_blocking=True)
            x = (sc.mfcc(augment(x, noise)) - mean) / std
            loss = loss_fn(model(x.unsqueeze(1)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        acc = accuracy(model, val_x, val_label)
        print(f"epoch {epoch:3d}  loss {loss.item():.3f}  val {acc:.2%}  {time.time() - start:.0f}s", flush=True)
        if acc > best:
            best = acc
            ckpt.update(model=model.state_dict(), epoch=epoch, val_acc=acc)
            torch.save(ckpt, args.out)

    model.load_state_dict(ckpt["model"])
    test_audio, test_label = sc.load_split("test", args.root)
    ckpt["test_acc"] = accuracy(model, features(test_audio, mean, std, dev), test_label)
    torch.save(ckpt, args.out)
    print(f"best epoch {ckpt['epoch']}  val {best:.2%}  test {ckpt['test_acc']:.2%}  -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="train ds-cnn on speech commands v2")
    p.add_argument("--width", type=int, default=172)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--root", type=Path, default=sc.ROOT)
    p.add_argument("--out", type=Path, default=REPO / "artifacts" / "ds_cnn.pt")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    train(args)
