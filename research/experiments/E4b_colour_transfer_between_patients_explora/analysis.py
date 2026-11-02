"""E4b: does training-time patient colour-transfer reduce the leakage gap and patient-ID decodability?

    arch -arm64 .venv/bin/python research/experiments/E4b_colour_transfer_between_patients_explora/analysis.py

Primary outcomes (v2 prediction, 2026-10-29): (1) leaky-vs-grouped leakage gap, same method as E1/E8, using
the 5 grouped-r0 folds (screen only -- no leaky-split runs for E4b, so the "leakage gap" here is approximated
via the grouped-fold BACC spread and compared directly to E1/E8's own grouped levels, NOT a leaky-vs-grouped
gap in the literal E1/E8 sense). (2) patient-ID probe on frozen embeddings from the resulting checkpoints,
identical method to E2.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
PKG_ROOT = HERE.parents[2]
sys.path.insert(0, str(PKG_ROOT))
from research.malaria import data, models  # noqa: E402
from research.malaria.stats import metrics  # noqa: E402
from research.experiments.E2_shortcut_evidence.analysis_embeddings import within_patient_probe  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
RUN = RESEARCH / "runs_kaggle" / "e4b-colour-transfer" / "runs"

ARM_CFG = {
    "E4b_e1": dict(dir="E4b_e1_colourtransfer", model=dict(model="danet", gate="softmax", dilation_mode="cascade", norm="instance")),
    "E4b_e8b": dict(dir="E4b_e8b_colourtransfer", model=dict(model="danet", gate="sigmoid", dilation_mode="parallel", norm="layer")),
    "E4b_e8c": dict(dir="E4b_e8c_colourtransfer", model=dict(model="danet", gate="sigmoid", dilation_mode="parallel", norm="instance")),
}
BASELINE_BACC = {"E4b_e1": 96.778, "E4b_e8b": None, "E4b_e8c": None}  # E8b/E8c filled from E8's results.json below


def find_ckpt(cfg, fold):
    base = RUN / cfg["dir"]
    hits = list(base.glob(f"grouped_r0f{fold}_t0*/best.pth"))
    assert len(hits) == 1, f"expected 1 checkpoint for fold {fold}, found {hits}"
    return hits[0]


def grouped_oof(cfg):
    frames = []
    for fold in range(5):
        d = pd.read_csv(find_ckpt(cfg, fold).parent / "test_predictions.csv")
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    assert d.idx.is_unique and len(d) == 27558, f"OOF incomplete: {len(d)} cells"
    return d


def to_tensor(root, cells, j):
    t = torch.from_numpy(data.read_rgb(root / cells.rel_path[j])).permute(2, 0, 1).float() / 255.0
    return torch.nn.functional.interpolate(t[None], size=(224, 224), mode="bilinear",
                                           align_corners=False, antialias=True)[0]


def release(model):
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def embed_and_probe(cfg, cells):
    all_idx, all_emb = [], []
    root = data.image_root()
    folds = pd.read_csv(RESEARCH / "splits" / "grouped_folds.csv")
    folds = folds[(folds.repeat == 0) & (folds.role == "test")]
    for fold in range(5):
        test_patients = set(folds[folds.fold == fold].patient)
        sub = cells[cells.patient.isin(test_patients)]
        model = models.build_model(cfg["model"]).to(DEVICE).eval()
        model.load_state_dict(torch.load(find_ckpt(cfg, fold), map_location=DEVICE))
        embs = []
        with torch.no_grad():
            for i in range(0, len(sub), 256):
                batch_idx = sub.index[i:i + 256]
                imgs = torch.stack([to_tensor(root, cells, j) for j in batch_idx])
                embs.append(model.embed(imgs.to(DEVICE)).cpu().numpy())
        all_idx.append(sub.index.values)
        all_emb.append(np.concatenate(embs))
        release(model)
        print(f"  {cfg['dir']} fold {fold}: {len(sub)} cells", flush=True)
    idx = np.concatenate(all_idx)
    emb = np.concatenate(all_emb)
    order = np.argsort(idx)
    idx, emb = idx[order], emb[order]
    assert (idx == cells.index.values).all()
    patient = cells.patient.values
    label = cells.label.values
    para_count = {p: label[patient == p].sum() for p in np.unique(patient)}
    infected_mask = np.array([para_count[p] > 0 for p in patient])
    return within_patient_probe(patient[infected_mask], emb[infected_mask])


def main():
    cells = pd.read_csv(RESEARCH / "splits" / "cells.csv", index_col="idx")
    e8_res = json.load(open(RESEARCH / "experiments" / "E8_paper_faithful_vs_code_faithful_danet" / "results.json"))
    BASELINE_BACC["E4b_e8b"] = e8_res["E8b (paper-faithful)"]["grouped_r0"]["BACC"]
    BASELINE_BACC["E4b_e8c"] = e8_res["E8c (paper-faithful, InstanceNorm)"]["grouped_r0"]["BACC"]
    baseline_probe = json.load(open(RESEARCH / "experiments" / "E2_shortcut_evidence" / "results_embeddings.json"))

    results = {}
    for arm, cfg in ARM_CFG.items():
        d = grouped_oof(cfg)
        m = metrics(d.label.values, d.p.values)
        probe = embed_and_probe(cfg, cells)
        base_key = {"E4b_e1": "E1", "E4b_e8b": "E8b", "E4b_e8c": "E8c"}[arm]
        results[arm] = dict(
            grouped_BACC=m["BACC"], baseline_grouped_BACC=BASELINE_BACC[arm],
            BACC_change=m["BACC"] - BASELINE_BACC[arm],
            patient_id_probe_lr_acc=probe["lr_acc"], patient_id_probe_chance=probe["chance"],
            baseline_probe_lr_acc=baseline_probe[base_key]["patient_id_probe_infected_patients"]["lr_acc"],
            probe_change=probe["lr_acc"] - baseline_probe[base_key]["patient_id_probe_infected_patients"]["lr_acc"])
        r = results[arm]
        print(f"{arm}: grouped BACC {r['grouped_BACC']:.2f} (baseline {r['baseline_grouped_BACC']:.2f}, "
             f"change {r['BACC_change']:+.2f}) | patient-ID probe {r['patient_id_probe_lr_acc']:.3f} "
             f"(baseline {r['baseline_probe_lr_acc']:.3f}, change {r['probe_change']:+.3f})")
    json.dump(results, open(HERE / "results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
