import numpy as np


def compute_known_mask(mask):
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def compute_missing_mask(known_mask):
    return np.where(known_mask > 0, 0, 255).astype(np.uint8)


def compute_inpaint_mask(known_mask):
    return compute_missing_mask(known_mask)
