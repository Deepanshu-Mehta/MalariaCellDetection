"""E5 step 1: for every MP-IDB Parasitized crop, build a parasite-pixel mask in the SAME crop frame as the
RGB image already saved by scripts/extract_mpidb.py. Reuses the cached Cellpose segmentation (no re-run).

    arch -arm64 .venv/bin/python research/experiments/E5_explanation_validity/build_parasite_masks.py

Writes research/data_external/mpidb_pf_cells/parasite_masks/{same relative path as the RGB crop}.png
(single-channel, 255 = parasite pixel, matching MARGIN/crop geometry from extract_mpidb.py exactly).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]            # research/
SRC = ROOT / "data_external" / "MP-IDB" / "Falciparum"
CELLS_DIR = ROOT / "data_external" / "mpidb_pf_cells"
MARGIN = 4                 # must match extract_mpidb.py's MARGIN exactly
PARASITE_INSIDE = 0.5      # must match extract_mpidb.py's threshold


def norm(name):
    return name.replace("-", "_")


def main():
    t = pd.read_csv(CELLS_DIR / "mpidb_cells.csv")
    para = t[t.label == "Parasitized"].copy()
    gt_files = {norm(g.name): g for g in (SRC / "gt").glob("*.jpg")}
    out_dir = CELLS_DIR / "parasite_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done, n_skipped = 0, 0
    for image, g in para.groupby("image"):
        cache = CELLS_DIR / "_masks" / f"{Path(image).stem}.npz"
        if not cache.exists():
            n_skipped += len(g); continue
        masks = np.load(cache)["masks"]
        gt = np.array(Image.open(gt_files[norm(image)]).convert("L")) > 127
        par_lab, n_par = ndi.label(gt)
        par_area = np.bincount(par_lab.ravel(), minlength=n_par + 1)
        h, w = masks.shape
        for row in g.itertuples():
            cid = row.cell
            ys, xs = np.where(masks == cid)
            if len(ys) == 0:
                n_skipped += 1; continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            ya, yb = max(y0 - MARGIN, 0), min(y1 + MARGIN, h)
            xa, xb = max(x0 - MARGIN, 0), min(x1 + MARGIN, w)
            cell_mask = masks[ya:yb, xa:xb] == cid
            # Keep only parasite components that were judged to belong to THIS cell during extraction
            # (>= PARASITE_INSIDE fraction inside the cell mask) -- matches extract_mpidb.py's own labelling.
            local_lab = par_lab[ya:yb, xa:xb]
            ids_in_crop = np.unique(local_lab[cell_mask & (local_lab > 0)])
            keep = np.zeros_like(cell_mask)
            for pid in ids_in_crop:
                frac = (local_lab == pid).sum() / max(par_area[pid], 1)
                if frac >= PARASITE_INSIDE:
                    keep |= (local_lab == pid)
            out_path = out_dir / row.file
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((keep * 255).astype(np.uint8)).save(out_path)
            n_done += 1
    print(f"built {n_done} parasite-mask crops, {n_skipped} skipped (no cache / no object)")


if __name__ == "__main__":
    main()
