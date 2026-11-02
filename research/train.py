"""Train and evaluate one model on one split. Every experiment is a JSON config in research/configs/.

    python -m research.train --config research/configs/E1_danet.json --split grouped --repeat 0 --fold 0
    python -m research.train --config research/configs/E1_danet.json --split leaky --seed 42
    python -m research.train --config ... --set epochs=2 data_frac=0.05      # smoke test

Outputs go to research/runs/<config name>/<split tag>/:
    config.json, train_log.csv, best.pth, last.pth (resume), test_predictions.csv, metrics.json
The script asserts that train, val and test patients are disjoint for every grouped / m3 split.
"""
import argparse
import io
import json
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from research.malaria import colour, data, metrics, splits
from research.malaria.models import build_model, count_params

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

DEFAULTS = dict(
    name="unnamed", model="danet", base_channels=8, gate="softmax", dilation_mode="cascade", norm="instance",
    pretrained=True, img_size=224, batch_size=64, epochs=50,
    optimizer="rmsprop_onecycle", max_lr=0.001, weight_decay=0.0,
    rotation=15, flip_p=0.2, colour_jitter=None,          # e.g. [0.1, 0.1, 0.5, 0.1] = M3 hypothesis
    label_mode="real", label_draw=0,                      # 'random_patient' = E3 permutation null
    colour_transfer=False, colour_transfer_prob=0.5,      # E4b: Reinhard-recolour training cells to a
                                                           # random other TRAINING patient's stain (2026-10-29)
    train_seed=0, data_frac=1.0, num_workers=2, max_hours=11.5,
)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_value(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def load_split(cells, split, repeat=0, fold=0, seed=42):
    """Returns (role Series aligned with cells: train / val / test, tag)."""
    sdir = ROOT / "splits"
    if split == "leaky":
        name = "leaky_splits.csv" if seed in splits.LEAKY_SEEDS else "leaky_splits_replication.csv"
        role = pd.read_csv(sdir / name, index_col="idx")[f"seed{seed}"]
        return role, f"leaky_s{seed}"
    if split in ("grouped", "session"):
        f = pd.read_csv(sdir / f"{split}_folds.csv")
        f = f[(f.repeat == repeat) & (f.fold == fold)]
        tag = f"{split}_r{repeat}f{fold}"
    elif split == "m3":
        f = pd.read_csv(sdir / "m3_splits.csv")
        f = f[f.seed == seed]
        tag = f"m3_s{seed}"
    else:
        raise ValueError(split)
    by_role = f.groupby("role").patient.apply(set)
    splits.assert_disjoint(by_role["train"] | by_role["val"], by_role["test"])
    splits.assert_disjoint(by_role["train"], by_role["val"])
    role = cells.patient.map(dict(zip(f.patient, f.role)))
    assert role.notna().all(), "cells without a split role"
    if split == "session":                            # E10: no imaging date may cross train, val and test
        date = cells.rel_path.str.extract(r"IMG_(\d{8})")[0]
        by_date = {r: set(date[role == r]) for r in ("train", "val", "test")}
        splits.assert_disjoint(by_date["train"] | by_date["val"], by_date["test"])
        splits.assert_disjoint(by_date["train"], by_date["val"])
    return role, tag


def random_patient_labels(cells, draw):
    """E3: one random label per patient, balanced so that ~50% of cells are positive."""
    rng = np.random.default_rng(1000 + draw)
    n = cells.groupby("patient").size()
    order = rng.permutation(n.index.values)
    cum = n[order].cumsum()
    pos = set(order[cum.values <= cum.iloc[-1] / 2])
    return cells.patient.isin(pos).astype(int)


class CellDataset(Dataset):
    """`colour_transfer`: optional (prob, patient_of_idx, patient_stats, rng) tuple. If given, each __getitem__
    call recolours its cell (Reinhard LAB, foreground-only) to a random OTHER patient's stats with probability
    `prob`, before the usual transform. Used only for E4b's training-time augmentation."""
    def __init__(self, idx, y, cache, tf, colour_transfer=None):
        self.idx, self.y, self.cache, self.tf = idx, y, cache, tf
        self.colour_transfer = colour_transfer

    def __len__(self):
        return len(self.idx)

    def _maybe_transfer(self, img_chw, i):
        prob, patient_of, stats, rng = self.colour_transfer
        if rng.random() >= prob:
            return img_chw
        src_p = patient_of[i]
        others = [p for p in stats if p != src_p]
        tgt_p = others[rng.integers(len(others))]
        arr = img_chw.permute(1, 2, 0).numpy()               # CHW uint8 tensor -> HWC uint8 array
        fg = data.foreground_mask(arr)
        if fg.sum() < 10 or src_p not in stats:
            return img_chw
        out = colour.reinhard_transfer(arr, stats[src_p], stats[tgt_p], fg)
        return torch.from_numpy(out).permute(2, 0, 1)

    def __getitem__(self, i):
        img = self.cache[self.idx[i]]
        if self.colour_transfer is not None:
            img = self._maybe_transfer(img, self.idx[i])
        return self.tf(img), int(self.y[i])


def transforms(cfg):
    s = cfg["img_size"]
    aug = [v2.Resize((s, s), antialias=True), v2.RandomRotation(cfg["rotation"]),
           v2.RandomHorizontalFlip(cfg["flip_p"]), v2.RandomVerticalFlip(cfg["flip_p"])]
    if cfg["colour_jitter"]:
        b, c, sat, h = cfg["colour_jitter"]
        aug.append(v2.ColorJitter(brightness=b, contrast=c, saturation=sat, hue=h))
    to_float = v2.ToDtype(torch.float32, scale=True)
    return v2.Compose(aug + [to_float]), v2.Compose([v2.Resize((s, s), antialias=True), to_float])


def make_optimizer(model, cfg, steps_per_epoch):
    if cfg["optimizer"] == "rmsprop_onecycle":       # frozen M2 baseline: momentum 0, no momentum cycling
        opt = torch.optim.RMSprop(model.parameters(), lr=cfg["max_lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["max_lr"], epochs=cfg["epochs"],
                                                    steps_per_epoch=steps_per_epoch, cycle_momentum=False)
    elif cfg["optimizer"] == "adamw_cosine":         # ResNet-18 reference fine-tuning
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["max_lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["max_lr"], epochs=cfg["epochs"],
                                                    steps_per_epoch=steps_per_epoch, pct_start=0.1,
                                                    anneal_strategy="cos")
    else:
        raise ValueError(cfg["optimizer"])
    return opt, sched


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logp = model(x)
        loss_sum += F.nll_loss(logp, y, reduction="sum").item()
        ys.append(y.cpu()); ps.append(logp[:, 1].exp().cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return y, p, loss_sum / max(len(y), 1)


def train(model, loaders, cfg, out):
    opt, sched = make_optimizer(model, cfg, len(loaders["train"]))
    start, hist, best = 0, [], float("inf")
    last = out / "last.pth"
    cfg_fingerprint = json.dumps(cfg, sort_keys=True)
    if last.exists():
        ck = torch.load(last, map_location=DEVICE, weights_only=False)
        # A directory collision (two logically different runs resolving to the same `out`) must fail loudly,
        # not silently resume one run's weights into another's bookkeeping. Bit us once in E3 (2026-10-29):
        # label_draw wasn't part of the output path, so draw 1 "resumed" draw 0's finished checkpoint and
        # overwrote its metrics with an untrained-for-draw-1 evaluation.
        if ck.get("cfg_fingerprint") not in (None, cfg_fingerprint):
            raise RuntimeError(f"checkpoint at {last} belongs to a different config than this run's -- "
                              f"output path collision, refusing to resume into it. Delete {out} if this run "
                              f"should start fresh, or fix the path so distinct configs don't share `out`.")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, hist, best = ck["epoch"] + 1, ck["hist"], ck["best"]
        torch.set_rng_state(ck["rng"].cpu())
        print(f"resumed from epoch {start}")
    t_run = time.time()
    for ep in range(start, cfg["epochs"]):
        t0 = time.time()
        model.train()
        n, loss_sum, correct = 0, 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logp = model(x)
            loss = F.nll_loss(logp, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep + 1}")
            loss.backward()
            opt.step(); sched.step()
            n += len(y); loss_sum += loss.item() * len(y); correct += (logp.argmax(1) == y).sum().item()
        yv, pv, val_loss = predict(model, loaders["val"])
        rec = dict(epoch=ep + 1, train_loss=loss_sum / n, train_acc=correct / n, val_loss=val_loss,
                   val_acc=float(((pv >= 0.5) == yv).mean()), lr=sched.get_last_lr()[0], sec=time.time() - t0)
        hist.append(rec)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), out / "best.pth")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep,
                        hist=hist, best=best, rng=torch.get_rng_state(), cfg_fingerprint=cfg_fingerprint), last)
        pd.DataFrame(hist).to_csv(out / "train_log.csv", index=False)
        print(f"ep {ep + 1:3d}/{cfg['epochs']} | train {rec['train_loss']:.4f}/{rec['train_acc']:.4f} | "
              f"val {val_loss:.4f}/{rec['val_acc']:.4f} | {rec['sec']:.0f}s" + ("  *best*" if best == val_loss else ""),
              flush=True)
        if (time.time() - t_run) / 3600 > cfg["max_hours"] and ep + 1 < cfg["epochs"]:
            print("time budget reached; rerun the same command to resume")
            return pd.DataFrame(hist), False
    return pd.DataFrame(hist), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["grouped", "session", "leaky", "m3"], required=True)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42, help="leaky / m3 split seed")
    ap.add_argument("--out-root", default=str(ROOT / "runs"))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (JSON values)")
    args = ap.parse_args()

    cfg = {**DEFAULTS, **json.load(open(args.config))}
    for kv in args.set:
        k, v = kv.split("=", 1)
        assert k in DEFAULTS, f"unknown config key {k}"
        cfg[k] = parse_value(v)
    seed_everything(cfg["train_seed"])

    cells = pd.read_csv(ROOT / "splits" / "cells.csv", index_col="idx")
    role, tag = load_split(cells, args.split, args.repeat, args.fold, args.seed)
    labels = cells.label if cfg["label_mode"] == "real" else random_patient_labels(cells, cfg["label_draw"])
    # Any --set override not already reflected in (config name, split tag, train_seed) must still make the
    # run's output path unique, or two runs that differ only in e.g. label_draw silently collide: the second
    # run finds the first run's finished last.pth, "resumes" into an empty epoch range, and evaluates an
    # untrained-for-this-config model against its own (different) labels. Bit us once in E3 (2026-10-29).
    extra = "_".join(sorted(kv.replace("=", "") for kv in args.set if not kv.startswith("train_seed=")))
    tag_full = f"{tag}_t{cfg['train_seed']}" + (f"_{extra}" if extra else "")
    out = Path(args.out_root) / cfg["name"] / tag_full
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(cfg=cfg, split=args.split, repeat=args.repeat, fold=args.fold, seed=args.seed, tag=tag,
                   device=str(DEVICE), torch=torch.__version__), open(out / "config.json", "w"), indent=1)

    use = cells.index.values
    if cfg["data_frac"] < 1:
        use = cells.groupby(role.values, group_keys=False).sample(frac=cfg["data_frac"],
                                                                 random_state=cfg["train_seed"]).index.values
    t0 = time.time()
    packed = os.environ.get("MALARIA_PACKED") or (next(Path("/kaggle/input").rglob("cells_png.pkl"), None)
                                                  if Path("/kaggle/input").exists() else None)
    if packed:                                        # one file instead of 27k small reads (see scripts/pack_images.py)
        blob = pickle.load(open(packed, "rb"))
        assert blob["rel_path"] == cells.rel_path.tolist(), "packed images do not match cells.csv"
        with ThreadPoolExecutor(8) as ex:
            imgs = list(ex.map(lambda i: np.asarray(Image.open(io.BytesIO(blob["png"][i])).convert("RGB")).copy(), use))
        del blob
    else:
        root = data.image_root()
        missing = [p for p in cells.rel_path[use] if not (root / p).exists()]
        assert not missing, f"{len(missing)} images missing under {root}, e.g. {missing[:3]}"
        with ThreadPoolExecutor(8) as ex:
            imgs = list(ex.map(lambda i: data.read_rgb(root / cells.rel_path[i]), use))
    cache = {i: torch.from_numpy(im).permute(2, 0, 1) for i, im in zip(use, imgs)}
    print(f"{tag}: decoded {len(cache)} images in {time.time() - t0:.0f}s on {DEVICE}")

    train_colour_transfer = None
    if cfg["colour_transfer"]:
        # Per-TRAINING-patient LAB stats only, from the images already decoded above -- never touches a
        # val/test-patient image, so no colour information crosses the train/test boundary.
        train_idx_for_stats = np.array([i for i in use if role[i] == "train"])
        img_by_idx = dict(zip(use, imgs))
        stats = {}
        for p, g_df in cells.loc[train_idx_for_stats].groupby("patient"):
            sample = g_df.sample(min(len(g_df), 10), random_state=cfg["train_seed"]).index.values
            means, stds = [], []
            for i in sample:
                arr = img_by_idx[i]
                fg = data.foreground_mask(arr)
                if fg.sum() >= 10:
                    m, s = colour.lab_stats(arr, fg)
                    means.append(m); stds.append(s)
            if means:
                stats[p] = (np.mean(means, axis=0), np.mean(stds, axis=0))
        patient_of = dict(zip(train_idx_for_stats, cells.patient[train_idx_for_stats]))
        # NB: with num_workers > 0, dataloader workers may share correlated RNG state (fork-inherited); fine
        # for this exploratory augmentation, not relied on for a precise statistical claim about randomness.
        train_colour_transfer = (cfg["colour_transfer_prob"], patient_of, stats, np.random.default_rng(cfg["train_seed"]))
        print(f"colour_transfer: {len(stats)} training patients with usable stats, prob={cfg['colour_transfer_prob']}")

    train_tf, eval_tf = transforms(cfg)
    loaders = {}
    for r in ["train", "val", "test"]:
        idx = np.array([i for i in use if role[i] == r])
        g = torch.Generator().manual_seed(cfg["train_seed"])
        ds = CellDataset(idx, labels[idx].values, cache, train_tf if r == "train" else eval_tf,
                         colour_transfer=train_colour_transfer if r == "train" else None)
        loaders[r] = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=r == "train", generator=g,
                                num_workers=cfg["num_workers"], pin_memory=DEVICE.type == "cuda",
                                persistent_workers=cfg["num_workers"] > 0)
    print({r: len(l.dataset) for r, l in loaders.items()})

    model = build_model(cfg).to(DEVICE)
    t_train = time.time()
    hist, finished = train(model, loaders, cfg, out)
    if not finished:
        return
    model.load_state_dict(torch.load(out / "best.pth", map_location=DEVICE))
    y, p, test_loss = predict(model, loaders["test"])
    test_idx = loaders["test"].dataset.idx
    pred = pd.DataFrame(dict(idx=test_idx, patient=cells.patient[test_idx].values, label=y, p=p))
    pred.to_csv(out / "test_predictions.csv", index=False)
    res = dict(cell=metrics.binary_metrics(y, p), patient=metrics.patient_macro(pred), test_loss=test_loss,
               best_epoch=int(hist.loc[hist.val_loss.idxmin(), "epoch"]), params=count_params(model),
               train_hours=(time.time() - t_train) / 3600,
               sec_per_epoch=float(hist.sec.median()))
    json.dump(res, open(out / "metrics.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
