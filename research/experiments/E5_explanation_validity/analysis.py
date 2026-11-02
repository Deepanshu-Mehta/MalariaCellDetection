"""E5: does DANet's Grad-CAM land on the parasite? MP-IDB expert masks, local, no Kaggle GPU needed.

    arch -arm64 .venv/bin/python research/experiments/E5_explanation_validity/analysis.py

Run on every Parasitized MP-IDB crop with a usable parasite mask (from build_parasite_masks.py), for three
arms (E1, E8b, E8c; grouped repeat-0 fold-0 checkpoints -- these never trained on MP-IDB at all, so there is
no leakage concern specific to this analysis).

Metrics:
  pointing game   does the CAM's single peak pixel (after a small dilation tolerance) fall inside the mask?
  energy ratio    fraction of total CAM energy inside the mask, vs the mask's own area fraction (chance)
  inpaint flip    does replacing the parasite region (resp. a random same-size foreground region) with the
                  cell's own mean colour flip the predicted class?
  sanity check    cascading layer randomisation (Adebayo 2018): does the CAM change once the model's last
                  layer is randomised? (cheap single-layer version, not the full cascade)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]            # research/
PKG_ROOT = HERE.parents[2]            # cv root, for `import research...`
sys.path.insert(0, str(PKG_ROOT))
from research.malaria import models  # noqa: E402
from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CELLS_DIR = RESEARCH / "data_external" / "mpidb_pf_cells"
DILATE_PX = 6             # pointing-game tolerance: parasites are tiny relative to the final 7x7 feature map


def load_model(cfg):
    m = models.build_model(cfg["model"]).to(DEVICE).eval()
    m.load_state_dict(torch.load(ckpt_path(cfg, fold=0), map_location=DEVICE))
    return m


def to_tensor(arr):
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return F.interpolate(t[None], size=(224, 224), mode="bilinear", align_corners=False, antialias=True)[0]


def grad_cam_v2(model, x):
    """Grad-CAM on the last DAB block's output: forward hook for activations, backward hook for their gradient."""
    store = {}
    layer = model.features[-1]
    def fwd(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))
    h = layer.register_forward_hook(fwd)
    model.zero_grad()
    logp = model(x)
    cls = logp.argmax(1)
    logp[0, cls].backward()
    h.remove()
    w = store["grad"].mean((2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / cam.max().clamp_min(1e-8)
    return cam.detach().cpu().numpy(), int(cls.item()), float(logp[0, cls].exp().item())


def dilate(mask, px):
    from scipy import ndimage as ndi
    return ndi.binary_dilation(mask, iterations=px) if px > 0 else mask


def pointing_game(cam, mask):
    py, px = np.unravel_index(cam.argmax(), cam.shape)
    return bool(dilate(mask, DILATE_PX)[py, px])


def energy_ratio(cam, mask):
    total = cam.sum()
    if total <= 0:
        return 0.0, mask.mean()
    return float(cam[mask].sum() / total), float(mask.mean())


def inpaint_region(img, region, fill):
    out = img.copy()
    out[region] = fill
    return out


def main():
    t = pd.read_csv(CELLS_DIR / "mpidb_cells.csv")
    para = t[t.label == "Parasitized"].reset_index(drop=True)

    rng = np.random.default_rng(0)
    results = {}
    for arm, cfg in ARMS.items():
        model = load_model(cfg)
        rows = []
        for r in para.itertuples():
            img = np.array(Image.open(CELLS_DIR / r.file).convert("RGB"))
            mask = np.array(Image.open(CELLS_DIR / "parasite_masks" / r.file)) > 0
            if mask.sum() < 4:
                continue
            fg = img.max(axis=2) > 0
            x = to_tensor(img)[None].to(DEVICE)
            cam, pred_cls, pred_p = grad_cam_v2(model, x)
            mask_224 = np.array(Image.fromarray((mask * 255).astype(np.uint8))
                               .resize((224, 224), resample=__import__("PIL").Image.NEAREST)) > 0
            pg = pointing_game(cam, mask_224)
            energy, area_frac = energy_ratio(cam, mask_224)

            # inpaint: parasite region -> cell's own foreground mean colour (in-distribution fill)
            fill = img[fg].mean(0) if fg.sum() else img.mean((0, 1))
            para_inpaint = inpaint_region(img, mask, fill)
            with torch.no_grad():
                p_para = float(torch.exp(model(to_tensor(para_inpaint)[None].to(DEVICE)))[0, 1])
            # random same-size foreground region, not overlapping the parasite mask
            ys, xs = np.where(fg & ~mask)
            rand_mask = np.zeros_like(mask)
            if len(ys):
                k = min(mask.sum(), len(ys))
                sel = rng.choice(len(ys), k, replace=False)
                rand_mask[ys[sel], xs[sel]] = True
            rand_inpaint = inpaint_region(img, rand_mask, fill)
            with torch.no_grad():
                p_rand = float(torch.exp(model(to_tensor(rand_inpaint)[None].to(DEVICE)))[0, 1])
            orig_pred = pred_cls
            rows.append(dict(file=r.file, pointing_game=pg, energy=energy, area_frac=area_frac,
                             orig_pred=orig_pred, flip_parasite_inpaint=int((p_para >= 0.5) != (orig_pred == 1)),
                             flip_random_inpaint=int((p_rand >= 0.5) != (orig_pred == 1))))
        d = pd.DataFrame(rows)
        d.to_csv(HERE / f"results_{arm}.csv", index=False)
        tp = d[d.orig_pred == 1]            # true positives only: the prediction is specifically about
                                             # cells the model itself calls Parasitized (all ground truth here
                                             # is Parasitized, by construction of `para`)

        # sanity check: randomise the last block's weights, does the CAM change?
        sample_idx = rng.choice(len(para), min(20, len(para)), replace=False)
        cam_diffs = []
        for i in sample_idx:
            r = para.iloc[i]
            img = np.array(Image.open(CELLS_DIR / r.file).convert("RGB"))
            x = to_tensor(img)[None].to(DEVICE)
            cam1, _, _ = grad_cam_v2(model, x)
            import copy
            model2 = copy.deepcopy(model)
            with torch.no_grad():
                for p in model2.features[-1].parameters():
                    p.normal_(0, p.std() if p.numel() > 1 else 0.1)
            cam2, _, _ = grad_cam_v2(model2, x)
            if cam1.std() == 0 or cam2.std() == 0:
                continue  # constant CAM (e.g. an unlucky ReLU-killing reinit) -- corrcoef is undefined, skip
            cam_diffs.append(1 - np.corrcoef(cam1.ravel(), cam2.ravel())[0, 1])

        results[arm] = dict(
            n_total=len(d), n_true_positive=len(tp), accuracy_on_parasitized=float((d.orig_pred == 1).mean()),
            pointing_game_rate=float(tp.pointing_game.mean()) if len(tp) else None,
            chance_area_frac=float(tp.area_frac.mean()) if len(tp) else None,
            energy_mean=float(tp.energy.mean()) if len(tp) else None,
            energy_minus_chance=float((tp.energy - tp.area_frac).mean()) if len(tp) else None,
            flip_rate_parasite_inpaint=float(tp.flip_parasite_inpaint.mean()) if len(tp) else None,
            flip_rate_random_inpaint=float(tp.flip_random_inpaint.mean()) if len(tp) else None,
            sanity_check_mean_1_minus_corr=float(np.mean(cam_diffs)) if cam_diffs else None,
            sanity_check_n_valid=len(cam_diffs))
        print(f"{arm}: n_tp={len(tp)}/{len(d)} (acc on parasitized {results[arm]['accuracy_on_parasitized']:.3f}) "
             f"pointing_game={results[arm]['pointing_game_rate']:.3f} "
             f"(chance~{results[arm]['chance_area_frac']:.3f} by area) energy={results[arm]['energy_mean']:.3f} "
             f"(+{results[arm]['energy_minus_chance']:.3f} vs chance) "
             f"flip_para={results[arm]['flip_rate_parasite_inpaint']:.3f} "
             f"flip_rand={results[arm]['flip_rate_random_inpaint']:.3f} "
             f"sanity(1-corr)={results[arm]['sanity_check_mean_1_minus_corr']:.3f}")

    json.dump(results, open(HERE / "results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
