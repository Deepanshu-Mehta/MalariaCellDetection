"""Recomputes the numbers behind the 2026-11-02 corrections to E4b, E5 and E9."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from research.malaria.stats import load_preds, paired_diff  # noqa: E402

R = ROOT / "research"
O = {}
d = paired_diff(load_preds("E4b_e8b_colourtransfer", "leaky_s42"), load_preds("E8_paper_faithful", "leaky_s42"), B=2000)
O["E4b_E8b_leaky_s42_paired_BACC"] = {k: float(v) for k, v in d.items() if k in ("diff", "lo", "hi")}

e = load_preds("E1_danet", "grouped")
e["rep"] = e.tag.str.extract(r"grouped_r(\d)f")[0].astype(int)
O["E9_DANet_minus_probe_by_repeat"] = {}
for bb in ("dinov2_s", "dinobloom_s", "reddino_s"):
    pr = pd.read_csv(R / f"experiments/E9_foundation_model_probes/oof_predictions_{bb}.csv")
    for rep in (0, 1, 2):
        o = paired_diff(e[e.rep == rep][["idx", "patient", "label", "p"]], pr, B=1000)
        O["E9_DANet_minus_probe_by_repeat"][f"{bb}_r{rep}"] = [round(float(o[k]), 3) for k in ("diff", "lo", "hi")]

C = R / "data_external/mpidb_pf_cells"
rows = []
for f in pd.read_csv(R / "experiments/E5_explanation_validity/results_E1.csv").file:
    img = np.array(Image.open(C / f).convert("RGB").resize((224, 224), Image.BILINEAR))
    m = np.array(Image.fromarray((np.array(Image.open(C / "parasite_masks" / f)) > 0).astype(np.uint8) * 255).resize((224, 224), Image.NEAREST)) > 127
    dm = ndi.binary_dilation(m, iterations=6)
    fg = img.max(2) > 0
    core = ndi.binary_erosion(fg, iterations=8)
    g = np.where(core, ndi.gaussian_filter(img.min(2).astype(float), 3), 1e9)
    py, px = np.unravel_index(g.argmin(), g.shape)
    rows.append(dict(file=f, centre=dm[112, 112], darkest=dm[py, px], dil_in_fg=dm.sum() / max(fg.sum(), 1)))
o = pd.DataFrame(rows)
O["E5_true_positive_hit_rates"] = {}
for arm in ("E1", "E8b", "E8c"):
    x = pd.read_csv(R / f"experiments/E5_explanation_validity/results_{arm}.csv").merge(o, on="file")
    t = x[x.orig_pred == 1]
    O["E5_true_positive_hit_rates"][arm] = dict(n=len(t), cam=float(t.pointing_game.mean()), centre=float(t.centre.mean()),
                                                darkest_pixel=float(t.darkest.mean()), dilated_mask_share_of_fg=float(t.dil_in_fg.mean()))
json.dump(O, open(HERE / "verify_corrections.json", "w"), indent=1)
print(json.dumps(O, indent=1))
