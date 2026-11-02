"""E7: cleanlab confident-learning audit of NIH label quality, using E1's pooled grouped OOF predictions.

    arch -arm64 .venv/bin/python research/experiments/E7_label_noise_audit/analysis.py

Uses E1 grouped repeat 0 (every cell scored exactly once, by a model that never trained on it -- the
pre-condition cleanlab needs for its predicted-probability input to be valid out-of-sample). Flags the
cells cleanlab judges most likely mislabelled, cross-checks them against Fuhad 2020's own suspect-cell list
where file names allow a match, and writes a 100-crop AI pre-screen sheet for the top-flagged cells (same
pattern as the MP-IDB audit in E6's external-data protocol).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load_oof_repeat0():
    d = pd.concat([pd.read_csv(f) for f in ROOT.glob("runs_kaggle/*/runs/E1_danet/grouped_r0f*/test_predictions.csv")
                  if "smoke" not in str(f)], ignore_index=True)
    assert d.idx.is_unique and len(d) == 27558, f"expected 27558 unique OOF cells, got {len(d)}"
    return d.sort_values("idx").reset_index(drop=True)


def main():
    cells = pd.read_csv(ROOT / "splits" / "cells.csv", index_col="idx")
    d = load_oof_repeat0()
    pred_probs = np.c_[1 - d.p.values, d.p.values]          # columns = [P(uninfected), P(parasitized)]
    labels = d.label.values

    issue_idx = find_label_issues(labels, pred_probs, return_indices_ranked_by="self_confidence")
    flagged = d.iloc[issue_idx].copy()
    flagged["rel_path"] = cells.rel_path[flagged.idx].values
    flagged["patient"] = cells.patient[flagged.idx].values
    flagged.to_csv(HERE / "flagged_cells.csv", index=False)

    res = dict(n_cells=len(d), n_flagged=len(flagged), flagged_fraction=len(flagged) / len(d),
              flagged_by_label=flagged.label.value_counts().to_dict(),
              flagged_patients=int(flagged.patient.nunique()))

    # cross-check against Fuhad 2020's suspect-cell counts (647 parasitized + 750 uninfected removed, no
    # public per-file list was locatable in Phase-0 research, so this is a COUNT comparison, not a file-level
    # match)
    res["fuhad_2020_removed_for_comparison"] = dict(parasitized=647, uninfected=750, total=1397,
                                                     note="count-level comparison only; no file-level list found")

    top = flagged.head(100).reset_index(drop=True)
    top["audit_id"] = range(1, len(top) + 1)
    top[["audit_id", "idx", "rel_path", "label", "patient", "p"]].to_csv(HERE / "audit_top100.csv", index=False)

    root = ROOT.parent / "milestone2-baseline" / "data" / "cell_images"
    sz = 110
    for k in range((len(top) + 24) // 25):
        part = top.iloc[k * 25:(k + 1) * 25]
        sheet = Image.new("RGB", (5 * sz, 5 * (sz + 16)), "white")
        dr = ImageDraw.Draw(sheet)
        for i, r in enumerate(part.itertuples()):
            x, y = (i % 5) * sz, (i // 5) * (sz + 16)
            lbl = "Para" if r.label == 1 else "Uninf"
            dr.text((x + 2, y), f"#{r.audit_id} {lbl} p={r.p:.2f}", fill="black")
            sheet.paste(Image.open(root / r.rel_path).convert("RGB").resize((sz - 4, sz - 4)), (x + 2, y + 16))
        sheet.save(HERE / f"audit_top100_sheet_{k}.png")

    json.dump(res, open(HERE / "results.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
