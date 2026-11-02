"""Pack the 27,558 NIH PNGs (raw bytes, cells.csv order) into one file for fast loading on Kaggle.

Reading the extracted dataset on Kaggle costs ~19 min per session (many small files on a cold network mount);
one ~350 MB file reads in seconds and decodes in ~1 min.

    arch -arm64 .venv/bin/python -m research.scripts.pack_images      # -> research/kaggle/_stage/packed/
    arch -arm64 .venv/bin/kaggle datasets create -p research/kaggle/_stage/packed
"""
import json
import pickle
from pathlib import Path

import pandas as pd

from research.malaria import data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "_stage" / "packed"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "splits" / "cells.csv", index_col="idx")
    root = data.image_root()
    blobs = [(root / p).read_bytes() for p in cells.rel_path]
    with open(OUT / "cells_png.pkl", "wb") as f:
        pickle.dump(dict(rel_path=cells.rel_path.tolist(), png=blobs), f, protocol=5)
    json.dump({"title": "nih-malaria-cells-packed", "id": "deepanshum27/nih-malaria-cells-packed",
               "licenses": [{"name": "other"}]}, open(OUT / "dataset-metadata.json", "w"))
    print(f"{len(blobs)} images, {sum(map(len, blobs)) / 1e6:.0f} MB -> {OUT}")


if __name__ == "__main__":
    main()
