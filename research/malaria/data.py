"""NIH Malaria cell images: file index, patient IDs and image loading.

Patient IDs come from NIH's own mapping CSVs. Those files list 201 IDs because patient C47P8 was imaged
on two microscopes (`C47P8thin_Original_Motic`, `C47P8thinOriginalOlympusCX21`). Every split in this
project groups on the leading `C<number>` token instead, which gives 200 patients (150 infected + 50
healthy-only) and keeps both halves of C47P8 on the same side of every split.
"""
import os
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

NIH_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/"
CLASSES = ["Parasitized", "Uninfected"]
LABEL = {"Uninfected": 0, "Parasitized": 1}          # Parasitized = positive class (label 1)


def default_data_dir():
    return Path(os.environ.get("MALARIA_DATA", Path(__file__).resolve().parents[2] / "milestone2-baseline" / "data"))


def find_class_root(base):
    """First directory under `base` whose sub-folders are exactly Parasitized/ and Uninfected/
    (skips the nested duplicate copy in the Kaggle mirror)."""
    for root, dirs, _ in os.walk(base):
        if sorted(d for d in dirs if not d.startswith(".")) == CLASSES:
            return Path(root)
    return None


def _fetch(name, dest):
    if not dest.exists():
        print("downloading", name, "...")
        urllib.request.urlretrieve(NIH_URL + name, dest)
    return dest


def image_root(data_dir=None):
    data_dir = Path(data_dir or default_data_dir())
    for cand in [data_dir, Path("/kaggle/input")]:
        if cand.exists() and (root := find_class_root(cand)):
            return root
    data_dir.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(_fetch("cell_images.zip", data_dir / "cell_images.zip")).extractall(data_dir)
    return find_class_root(data_dir)


def patient_key(nih_id):
    """Grouping key: the leading `C<number>` token of an NIH patient ID (merges C47P8's two microscopes)."""
    m = re.match(r"C\d+", nih_id)
    if m is None:
        raise ValueError(f"unexpected NIH patient id: {nih_id!r}")
    return m.group(0)


def load_patient_map(data_dir=None):
    """{(class, filename): NIH patient id}. The CSVs are 'id,[list of files],,,,' with trailing commas,
    so they are parsed with a regex rather than csv/ast."""
    data_dir = Path(data_dir or default_data_dir())
    pmap = {}
    for cls in CLASSES:
        name = f"patientid_cellmapping_{cls.lower()}.csv"
        path = data_dir / name
        if not path.exists():
            kaggle = next(Path("/kaggle/input").rglob(name), None) if Path("/kaggle/input").exists() else None
            path = kaggle or _fetch(name, path)
        for line in open(path):
            nih_id = line.split(",", 1)[0].strip()
            for fn in re.findall(r"'([^']+\.png)'", line):
                pmap[(cls, fn)] = nih_id
    return pmap


def build_index(data_dir=None):
    """One row per cell: rel_path, cls, label, nih_id, patient (merged C-number), microscope_variant."""
    root = image_root(data_dir)
    pmap = load_patient_map(data_dir)
    rows = [(f"{cls}/{fn}", cls, LABEL[cls], pmap.get((cls, fn)))
            for cls in CLASSES for fn in sorted(os.listdir(root / cls)) if fn.endswith(".png")]
    df = pd.DataFrame(rows, columns=["rel_path", "cls", "label", "nih_id"])
    assert (df.cls.value_counts() == 13779).all(), "unexpected dataset contents"
    assert df.nih_id.notna().all(), "cells missing from the NIH patient map"
    df["patient"] = df.nih_id.map(patient_key)
    df.attrs["image_root"] = str(root)
    return df


def patient_table(df):
    """Per-patient counts and type: 'healthy' (no parasitized cells) or 'infected'."""
    t = df.groupby("patient").agg(n_cells=("label", "size"), n_para=("label", "sum"))
    t["n_uninf"] = t.n_cells - t.n_para
    t["para_frac"] = t.n_para / t.n_cells
    t["ptype"] = np.where(t.n_para == 0, "healthy", "infected")
    return t


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def load_images(df, workers=8):
    """Decode every image once (uint8 HxWx3 arrays, native size)."""
    root = Path(df.attrs["image_root"])
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda r: read_rgb(root / r), df.rel_path))


def foreground_mask(img, thr=10):
    """NIH crops are padded with pure black; foreground = any channel above `thr`."""
    return img.max(axis=2) > thr
