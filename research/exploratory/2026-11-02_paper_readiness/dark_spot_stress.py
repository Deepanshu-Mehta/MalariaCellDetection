"""Independent reimplementation of the dark-spot feature, plus stress tests and an E6c condition C diagnosis."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "research"
NIH = ROOT / "milestone2-baseline/data/cell_images"
MP = R / "data_external/mpidb_pf_cells"
SRC = R / "data_external/MP-IDB/Falciparum"
SCR = Path(__file__).parent
EROSION, SIGMA = 4, 1.5


def feature(arr):
    im = np.array(Image.fromarray(arr).resize((128, 128), Image.BILINEAR)).astype(float)
    fg = im.max(2) > 0
    core = ndi.binary_erosion(fg, iterations=EROSION)
    if core.sum() < 30:
        return np.nan
    g = ndi.gaussian_filter(im[..., 1] * fg, SIGMA) / np.maximum(ndi.gaussian_filter(fg.astype(float), SIGMA), 1e-3)
    med = np.median(g[core])
    return (med - g[core].min()) / max(med, 1)


def from_path(p):
    return feature(np.array(Image.open(p).convert("RGB")))


def many(paths):
    with ProcessPoolExecutor(6) as ex:
        return np.array(list(ex.map(from_path, paths, chunksize=200)))


def auc(y, s):
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    return (st.rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rates(y, s, t):
    y = np.asarray(y).astype(bool)
    q = np.asarray(s) >= t
    se, sp = 100 * q[y].mean(), 100 * (~q[~y]).mean()
    return dict(BACC=(se + sp) / 2, SE=se, SP=sp)


def best_thr(y, s):
    qs = np.quantile(s, np.linspace(.02, .98, 193))
    return qs[int(np.argmax([rates(y, s, q)["BACC"] for q in qs]))]


def main():
    sys.path.insert(0, str(ROOT))
    O = {}
    cells = pd.read_csv(R / "splits/cells.csv")
    cells["f"] = np.nan_to_num(many([NIH / p for p in cells.rel_path]))
    t = pd.read_csv(MP / "mpidb_cells.csv")
    m = t[t.label.notna()].reset_index(drop=True)
    m["y"] = (m.label == "Parasitized").astype(int)
    m["f"] = np.nan_to_num(many([MP / p for p in m.file]))
    cells.to_pickle(SCR / "nih_f.pkl")
    m.to_pickle(SCR / "mp_f.pkl")

    # A. Reproduce, then a fold-honest NIH threshold and a paired comparison with DANet
    th_all = best_thr(cells.label, cells.f)
    O["A_reproduce"] = dict(NIH_AUC=auc(cells.label, cells.f), thr=th_all, NIH=rates(cells.label, cells.f, th_all),
                            MP_AUC=auc(m.y, m.f), MP=rates(m.y, m.f, th_all))
    folds = pd.read_csv(R / "splits/grouped_folds.csv")
    folds = folds[(folds.repeat == 0) & (folds.role == "test")].set_index("patient").fold
    cells["fold"] = cells.patient.map(folds)
    pred_oof = pd.Series(False, index=cells.index)
    ths = []
    for k in range(5):
        tr, te = cells.fold != k, cells.fold == k
        th = best_thr(cells.label[tr], cells.f[tr])
        ths.append(float(th))
        pred_oof[te] = cells.f[te] >= th
    from research.experiments.E2_shortcut_evidence.extract_embeddings import ARMS, ckpt_path
    e = pd.concat([pd.read_csv(ckpt_path(ARMS["E1"], k).parent / "test_predictions.csv") for k in range(5)])
    d = cells.merge(e[["idx", "p"]], on="idx").assign(fpred=pred_oof.values)
    O["A_thresholds_by_fold"] = ths
    groups = [g.index.values for _, g in d.groupby("patient")]
    y, fp, dp, fs, ps = d.label.values == 1, d.fpred.values, d.p.values >= .5, d.f.values, d.p.values
    B = lambda yy, q: 50 * (q[yy].mean() + (~q[~yy]).mean())
    rng = np.random.default_rng(0)
    dB, dA = [], []
    for _ in range(2000):
        i = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        dB.append(B(y[i], dp[i]) - B(y[i], fp[i]))
        dA.append(auc(y[i], ps[i]) - auc(y[i], fs[i]))
    O["A_NIH_vs_DANet_patient_bootstrap"] = dict(
        feature_oof_BACC=B(y, fp), danet_BACC=B(y, dp), feature_AUC=auc(y, fs), danet_AUC=auc(y, ps),
        BACC_diff=[float(np.mean(dB)), *np.percentile(dB, [2.5, 97.5]).tolist()],
        AUC_diff=[float(np.mean(dA)), *np.percentile(dA, [2.5, 97.5]).tolist()],
        errors_feature=int((fp != y).sum()), errors_danet=int((dp != y).sum()))
    pinf = cells.groupby("patient").label.sum() > 0
    h = d[d.patient.isin(pinf.index[~pinf])]
    O["A_healthy_only_FP_pct"] = dict(feature=100 * h.fpred.mean(), danet=100 * (h.p >= .5).mean(), n=len(h))

    # B. Trained comparators on MP-IDB, including colour-transfer checkpoints
    S = R / "experiments/E6b_five_checkpoint_external_rescore/scores"
    arms, logit = ("E1", "E8b", "E8c"), lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    assert (m.file.values == pd.read_csv(R / "experiments/E6_external_test/mpidb_predictions.csv").file.values).all()
    comp = {}
    for c in "AB":
        L = np.array([logit(np.load(S / f"{c}_{a}_f{k}.npy")) for a in arms for k in range(5)])
        ens = 1 / (1 + np.exp(-L.mean(0)))
        comp[c] = dict(mean_single_AUC=float(np.mean([auc(m.y, l) for l in L])), ens15_AUC=auc(m.y, L.mean(0)),
                       ens15=rates(m.y, ens, .5))
    O["B_trained_on_MPIDB"] = comp

    # C. Exclusions the crop pipeline applied only to cells without parasite overlap
    rows = []
    for name, g in t[t.excluded.isin(["ambiguous_overlap", "leukocyte_like", "area_outlier"])].groupby("image"):
        img = np.array(Image.open(SRC / "img" / name).convert("RGB"))
        masks = np.load(MP / "_masks" / f"{Path(name).stem}.npz")["masks"]
        gtf = {p.name.replace("-", "_"): p for p in (SRC / "gt").glob("*.jpg")}[name.replace("-", "_")]
        gt = np.array(Image.open(gtf).convert("L")) > 127
        objs = ndi.find_objects(masks)
        for r in g.itertuples():
            sl = objs[r.cell - 1]
            ya, yb = max(sl[0].start - 4, 0), min(sl[0].stop + 4, img.shape[0])
            xa, xb = max(sl[1].start - 4, 0), min(sl[1].stop + 4, img.shape[1])
            keep = masks[ya:yb, xa:xb] == r.cell
            rows.append(dict(reason=r.excluded, f=feature((img[ya:yb, xa:xb] * keep[..., None]).astype(np.uint8)),
                             touches_gt=bool(gt[ya:yb, xa:xb][keep].any()), rel=r.area))
    x = pd.DataFrame(rows)
    x["fires"] = x.f >= th_all
    O["C_excluded_fire_rate"] = x.groupby(["reason", "touches_gt"]).fires.agg(["size", "mean"]).reset_index().to_dict("records")
    clean = x[(x.reason == "leukocyte_like") | ((x.reason == "ambiguous_overlap") & ~x.touches_gt)]
    worst = rates(np.r_[m.y, np.zeros(len(clean))], np.r_[m.f, np.nan_to_num(clean.f)], th_all)
    allneg = x[x.reason != "area_outlier"]
    O["C_SP_if_added_as_negatives"] = dict(clean_only=worst, n_clean=len(clean),
                                           all_nonoutlier=rates(np.r_[m.y, np.zeros(len(allneg))], np.r_[m.f, np.nan_to_num(allneg.f)], th_all))

    # D. Contact sheet: highest-scoring Uninfected and lowest-scoring Parasitized crops
    sheet = []
    for sub in (m[m.y == 0].nlargest(30, "f"), m[m.y == 1].nsmallest(30, "f")):
        ims = [np.array(Image.open(MP / p).convert("RGB").resize((96, 96))) for p in sub.file]
        sheet.append(np.vstack([np.hstack(ims[i:i + 15]) for i in (0, 15)]))
    Image.fromarray(np.vstack(sheet)).resize((1440, 768), Image.NEAREST).save(SCR / "feature_extremes.png")
    O["D_uninfected_over_thr"] = int(((m.y == 0) & (m.f >= th_all)).sum())
    O["D_stage_note"] = "see protocol: MP-IDB is mostly rings"

    # E. Condition C diagnosis: scale factors the registered normalisation actually applied
    from research.experiments.E6b_five_checkpoint_external_rescore import score_external as se
    from research.malaria import colour, data
    src = se.field_stats(se.scored_table())
    per = {}
    for r in se.scored_table().sample(4000, random_state=0).itertuples():
        arr = np.array(Image.open(MP / r.file).convert("RGB"))
        fg = data.foreground_mask(arr)
        if fg.sum() >= 10:
            per.setdefault(r.image, []).append(colour.lab_stats(arr, fg)[1])
    sf = np.array([v[1] for v in src.values()])
    within = np.array([np.mean(v, axis=0) for k, v in per.items()])
    tgt = se.fold_targets()[0]
    O["E_condC"] = dict(target_mean=tgt[0].tolist(), target_std=tgt[1].tolist(),
                        source_field_mean=np.array([v[0] for v in src.values()]).mean(0).tolist(),
                        source_field_std=sf.mean(0).tolist(), source_within_cell_std=within.mean(0).tolist(),
                        applied_scale=(tgt[1] / sf.mean(0)).tolist(),
                        consistent_scale=(tgt[1] / within.mean(0)).tolist())
    json.dump(O, open(SCR / "dark_spot_stress.json", "w"), indent=1, default=float)
    print(json.dumps(O, indent=1, default=float))


if __name__ == "__main__":
    main()
