"""Box-to-crop rules shared by the MP-IDB rehearsal and BBBC041. BBBC041 has boxes, not masks.

ellipse (primary): the ellipse inscribed in the box, background zeroed. Uses no pixel values, so it has no failure mode.
otsu (sensitivity): Otsu on the green channel, cell on the dark side, largest component, holes filled.
Nothing here reads a label or a model score.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

MIN_MASK_FRAC = 0.2                 # below this the mask failed, and the unmasked box is kept and flagged


def otsu_mask(box_rgb):
    g = box_rgb[..., 1]
    _, dark = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = ndi.binary_opening(dark.astype(bool), iterations=2)
    lab, n = ndi.label(dark)
    if n == 0:
        return np.zeros(g.shape, bool)
    keep = lab == (1 + int(np.argmax(ndi.sum(dark, lab, index=range(1, n + 1)))))
    return ndi.binary_fill_holes(keep)


def ellipse_mask(h, w):
    yy, xx = np.mgrid[:h, :w]
    return ((yy - (h - 1) / 2) / (h / 2)) ** 2 + ((xx - (w - 1) / 2) / (w / 2)) ** 2 <= 1


def crop(image, r0, c0, r1, c1, rule="ellipse"):
    """Returns (crop with zeroed background, mask, mask_failed)."""
    r0, c0 = max(int(r0), 0), max(int(c0), 0)
    box = image[r0:int(r1), c0:int(c1)]
    if rule == "ellipse":
        mask = ellipse_mask(*box.shape[:2])
        return box * mask[..., None], mask, False
    mask = otsu_mask(box)
    if mask.mean() < MIN_MASK_FRAC:
        return box.copy(), np.ones(box.shape[:2], bool), True
    return box * mask[..., None], mask, False
