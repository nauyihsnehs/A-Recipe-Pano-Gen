import numpy as np


def create_equirectangular_canvas(width, height):
    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError("canvas width and height must be positive")
    if width != height * 2:
        raise ValueError("equirectangular canvas must use a 2:1 width:height ratio")

    panorama = np.zeros((height, width, 3), dtype=np.uint8)
    known_mask = np.zeros((height, width), dtype=np.uint8)

    return panorama, known_mask


def paste_projected_view(panorama, known_mask, projected, projection_mask):
    if panorama.shape[:2] != known_mask.shape:
        raise ValueError("panorama and known_mask dimensions do not match")
    if panorama.shape[:2] != projected.shape[:2]:
        raise ValueError("panorama and projected dimensions do not match")
    if known_mask.shape != projection_mask.shape:
        raise ValueError("known_mask and projection_mask dimensions do not match")

    output = panorama.copy()
    output_mask = np.maximum(known_mask, projection_mask)
    region = projection_mask > 0
    output[region] = projected[region]

    return output, output_mask
