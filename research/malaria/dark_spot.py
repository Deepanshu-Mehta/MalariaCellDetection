"""The frozen no-training dark-spot rule (BBBC041 candidate 1). Do not edit: its hash is registered in E11.

Input: an RGB uint8 crop with background set to zero. Score: (median - min) / median of the smoothed green channel
inside the eroded cell, at 128 px. A cell is called Parasitized when the score is at least THRESHOLD, the exact NIH maximiser.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

EROSION, SIGMA, SIZE = 4, 1.5, 128
THRESHOLD = 0.3047


def score(arr):
    im = np.array(Image.fromarray(arr).resize((SIZE, SIZE), Image.BILINEAR)).astype(float)
    fg = im.max(2) > 0
    core = ndi.binary_erosion(fg, iterations=EROSION)
    if core.sum() < 30:
        return np.nan
    g = ndi.gaussian_filter(im[..., 1] * fg, SIGMA) / np.maximum(ndi.gaussian_filter(fg.astype(float), SIGMA), 1e-3)
    med = np.median(g[core])
    return (med - g[core].min()) / max(med, 1)
