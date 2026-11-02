"""E5b: E5's Grad-CAM pointing game on 15 checkpoints, against no-model rules. Resume-safe, one CSV per checkpoint.

    caffeinate -i arch -arm64 .venv/bin/python -m research.experiments.E5b_five_checkpoint_explanations.run
    arch -arm64 .venv/bin/python -m research.experiments.E5b_five_checkpoint_explanations.run --analyse

E5's files are read, never written. Inpainting is left out: the registered question is the hit rate.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi

from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path
from research.experiments.E5_explanation_validity.analysis import DILATE_PX, energy_ratio, grad_cam_v2, pointing_game, to_tensor
from research.experiments.E6b_five_checkpoint_external_rescore.score_external import CELLS_DIR, fast_norms
from research.malaria import models

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
E5 = RESEARCH / "experiments" / "E5_explanation_validity"
CAMS = HERE / "per_checkpoint"
DEVICE = torch.device("mps")
B, FOLDS = 2000, range(5)


def table():
    t = pd.read_csv(CELLS_DIR / "mpidb_cells.csv")
    return t[t.label == "Parasitized"].reset_index(drop=True)


def load(file):
    img = np.array(Image.open(CELLS_DIR / file).convert("RGB"))
    mask = np.array(Image.open(CELLS_DIR / "parasite_masks" / file)) > 0
    if mask.sum() < 4:
        return None
    mask224 = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224), resample=Image.NEAREST)) > 0
    return img, mask224


def no_model_rules():
    """verify_corrections.py's rules. Fixed per crop, then scored on each checkpoint's true positives."""
    path = HERE / "no_model_rules.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for r in table().itertuples():
        got = load(r.file)
        if got is None:
            continue
        img = np.array(Image.fromarray(got[0]).resize((224, 224), Image.BILINEAR))
        dm = ndi.binary_dilation(got[1], iterations=DILATE_PX)
        fg = img.max(2) > 0
        core = ndi.binary_erosion(fg, iterations=8)
        g = np.where(core, ndi.gaussian_filter(img.min(2).astype(float), 3), 1e9)
        py, px = np.unravel_index(g.argmin(), g.shape)
        rows.append(dict(file=r.file, darkest=bool(dm[py, px]), centre=bool(dm[112, 112])))
    d = pd.DataFrame(rows)
    d.to_csv(path, index=False)
    return d


def run_cams():
    CAMS.mkdir(exist_ok=True)
    para = table()
    for arm, cfg in ARMS.items():
        for f in FOLDS:
            out = CAMS / f"{arm}_f{f}.csv"
            if out.exists():
                continue
            t0 = time.time()
            m = models.build_model(cfg["model"]).eval()
            m.load_state_dict(torch.load(ckpt_path(cfg, f), map_location="cpu"))
            m = fast_norms(m).to(DEVICE)
            rows = []
            for r in para.itertuples():
                got = load(r.file)
                if got is None:
                    continue
                img, mask = got
                cam, cls, _ = grad_cam_v2(m, to_tensor(img)[None].to(DEVICE))
                e, a = energy_ratio(cam, mask)
                rows.append(dict(file=r.file, pointing_game=pointing_game(cam, mask), energy=e, area_frac=a, orig_pred=cls))
            d = pd.DataFrame(rows)
            tmp = out.with_suffix(".tmp")
            d.to_csv(tmp, index=False)
            tmp.rename(out)
            del m
            gc.collect()
            torch.mps.empty_cache()
            print(f"{arm} f{f}: {len(d)} crops, {time.time() - t0:.0f}s", flush=True)


def hit_rates(d):
    tp = d[d.orig_pred == 1]
    return dict(n_tp=len(tp), cam=100 * tp.pointing_game.mean(), darkest=100 * tp.darkest.mean(), centre=100 * tp.centre.mean())


def analyse():
    rules = no_model_rules()
    per = {}
    for arm in ARMS:
        saved = pd.read_csv(E5 / f"results_{arm}.csv")
        now = pd.read_csv(CAMS / f"{arm}_f0.csv").merge(saved, on="file", suffixes=("", "_e5"))
        assert len(now) == len(saved)
        per[arm] = {f: pd.read_csv(CAMS / f"{arm}_f{f}.csv").merge(rules, on="file") for f in FOLDS}
        per[arm]["reproduction"] = dict(agree_hit=float((now.pointing_game == now.pointing_game_e5).mean()),
                                        agree_pred=float((now.orig_pred == now.orig_pred_e5).mean()),
                                        max_energy_diff=float((now.energy - now.energy_e5).abs().max()))
    O = dict(reproduction={a: per[a].pop("reproduction") for a in ARMS})
    files = per["E1"][0].file.values
    grid = {a: {f: per[a][f].set_index("file").loc[files] for f in FOLDS} for a in ARMS}
    rng = np.random.default_rng(0)
    draws = {a: [] for a in ARMS}
    for _ in range(B):
        c = rng.integers(0, len(files), len(files))
        fs = rng.integers(0, 5, 5)
        for a in ARMS:
            vals = []
            for f in fs:
                g = grid[a][f].iloc[c]
                tp = g.orig_pred.values == 1
                vals.append(100 * (g.darkest.values[tp].mean() - g.pointing_game.values[tp].mean()))
            draws[a].append(np.mean(vals))
    O["arms"] = {}
    wins = 0
    for a in ARMS:
        rows = [hit_rates(per[a][f]) for f in FOLDS]
        wins += sum(r["darkest"] > r["cam"] for r in rows)
        O["arms"][a] = dict(per_fold=rows, cam_mean=float(np.mean([r["cam"] for r in rows])),
                            cam_sd=float(np.std([r["cam"] for r in rows], ddof=1)),
                            darkest_mean=float(np.mean([r["darkest"] for r in rows])),
                            centre_range=[min(r["centre"] for r in rows), max(r["centre"] for r in rows)],
                            darkest_minus_cam=float(np.mean([r["darkest"] - r["cam"] for r in rows])),
                            ci=np.percentile(draws[a], [2.5, 97.5]).tolist())
    O["checkpoints_darkest_wins"] = wins
    O["rule_all_arms_ci_above_0"] = bool(all(O["arms"][a]["ci"][0] > 0 for a in ARMS))
    json.dump(O, open(HERE / "results.json", "w"), indent=1, default=float)
    print(json.dumps(O, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    if ap.parse_args().analyse:
        analyse()
    else:
        no_model_rules()
        run_cams()
