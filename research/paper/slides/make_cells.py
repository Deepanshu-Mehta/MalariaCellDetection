"""Round cell thumbnails for the title slide, drawn at random with a fixed seed.

    arch -arm64 .venv/bin/python -m research.paper.slides.make_cells
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from research.experiments.E6b_five_checkpoint_external_rescore.score_external import CELLS_DIR, scored_table
from research.malaria.data import image_root

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
S = 360


def disc(path):
    im = Image.open(path).convert("RGB").resize((S, S), Image.BILINEAR)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, S - 1, S - 1), fill=255)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(im, (0, 0), mask)
    ImageDraw.Draw(out).ellipse((4, 4, S - 5, S - 5), outline=(255, 255, 255, 230), width=8)
    return out


def main():
    rng = np.random.default_rng(20261103)
    pick = lambda s: s.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
    cells = pd.read_csv(R / "splits/cells.csv")
    mp = scored_table()
    lab = pd.read_csv(R / "experiments/E11_bbbc041_holdout/labels.csv")
    b41 = R / "data_external/bbbc041_crops/ellipse"
    paths = [image_root() / pick(cells[cells.label == 1].rel_path), image_root() / pick(cells[cells.label == 0].rel_path),
             CELLS_DIR / pick(mp[mp.binary_label == 1].file),
             b41 / f"{pick(lab[(lab.y == 1) & (lab.group == 'jpg')].crop_id)}.png",
             b41 / f"{pick(lab[(lab.y == 1) & (lab.group == 'png')].crop_id)}.png"]
    gap = 40
    strip = Image.new("RGBA", (len(paths) * S + (len(paths) - 1) * gap, S), (0, 0, 0, 0))
    for i, p in enumerate(paths):
        strip.alpha_composite(disc(p), (i * (S + gap), 0))
    strip.save(HERE / "img" / "title_cells.png")
    print("wrote title_cells.png")


if __name__ == "__main__":
    main()
