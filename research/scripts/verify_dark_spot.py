"""Check the frozen dark-spot module against its development numbers on NIH and MP-IDB.

    arch -arm64 .venv/bin/python -m research.scripts.verify_dark_spot
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as st

from research.malaria import dark_spot

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "research"
NIH = ROOT / "milestone2-baseline/data/cell_images"
MP = R / "data_external/mpidb_pf_cells"


def from_path(p):
    return dark_spot.score(np.array(Image.open(p).convert("RGB")))


def auc(y, s):
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    return (st.rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bacc(y, s, t):
    y = np.asarray(y).astype(bool)
    q = np.asarray(s) >= t
    return 50 * (q[y].mean() + (~q[~y]).mean())


def main():
    c = pd.read_csv(R / "splits/cells.csv")
    m = pd.read_csv(MP / "mpidb_cells.csv")
    m = m[m.label.notna()].reset_index(drop=True)
    with ProcessPoolExecutor(4) as ex:
        fn = np.array(list(ex.map(from_path, [NIH / p for p in c.rel_path], chunksize=200)))
        fm = np.array(list(ex.map(from_path, [MP / p for p in m.file], chunksize=200)))
    nan_n, nan_m = int(np.isnan(fn).sum()), int(np.isnan(fm).sum())
    fn, fm = np.nan_to_num(fn), np.nan_to_num(fm)
    y, ym = c.label.values == 1, (m.label == "Parasitized").values

    order = np.argsort(fn)
    fs, ys = fn[order], y[order]
    b = 50 * ((y.sum() - np.cumsum(ys)) / y.sum() + np.cumsum(~ys) / (~y).sum())
    i = int(np.argmax(b))
    exact = float((fs[i] + fs[i + 1]) / 2)
    print(json.dumps(dict(nan_nih=nan_n, nan_mp=nan_m, exact_maximiser=exact, frozen=dark_spot.THRESHOLD,
                          nih_auc=auc(y, fn), nih_bacc_frozen=bacc(y, fn, dark_spot.THRESHOLD), nih_bacc_exact=bacc(y, fn, exact),
                          mp_auc=auc(ym, fm), mp_bacc_frozen=bacc(ym, fm, dark_spot.THRESHOLD), mp_bacc_exact=bacc(ym, fm, exact),
                          mp_se=100 * (fm[ym] >= dark_spot.THRESHOLD).mean(), mp_sp=100 * (fm[~ym] < dark_spot.THRESHOLD).mean()), indent=1))


if __name__ == "__main__":
    main()
