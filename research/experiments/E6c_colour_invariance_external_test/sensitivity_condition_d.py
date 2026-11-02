"""Unregistered sensitivity for E6c: condition C rerun with source stats defined like the NIH targets."""
import json
from pathlib import Path

import numpy as np

from research.experiments.E6b_five_checkpoint_external_rescore.analysis import (
    METRICS, fast_metrics, hierarchical_boot, interval, load_scores)
from research.experiments.E6b_five_checkpoint_external_rescore.score_external import ARMS, N_FOLDS, scored_table

HERE = Path(__file__).resolve().parent
t = scored_table()
y, images = t.binary_label.values, t.image.values
S = {c: load_scores(c) for c in "AD"}
keys = [(c, a) for c in "AD" for a in ARMS]
boot = hierarchical_boot({(c, a): {f: S[c][(a, f)] for f in range(N_FOLDS)} for c, a in keys}, y, images, keys)
per = {c: {a: [fast_metrics(y, S[c][(a, f)]) for f in range(N_FOLDS)] for a in ARMS} for c in "AD"}
out = {"levels": {a: {m: float(np.mean([x[m] for x in per["D"][a]])) for m in METRICS} for a in ARMS}, "pooled_D_minus_A": {}}
for m in METRICS:
    d = np.mean([[per["D"][a][f][m] - per["A"][a][f][m] for f in range(N_FOLDS)] for a in ARMS], axis=0)
    out["pooled_D_minus_A"][m] = dict(mean=float(d.mean()), hier_ci=interval(np.mean([boot[("D", a)][m] - boot[("A", a)][m] for a in ARMS], axis=0)))
json.dump(out, open(HERE / "sensitivity_condition_d.json", "w"), indent=1)
print(json.dumps(out, indent=1))
