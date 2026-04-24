import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from camera import estimate_or_load_fov
from canvas import create_equirectangular_canvas, paste_projected_view
from mask import compute_missing_mask
from projection import (
    project_perspective_to_equirect,
    render_perspective_from_equirect,
    render_perspective_mask_from_equirect,
)


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    Image.fromarray(image).save(path)


def absolute_difference(image_a, image_b):
    diff = np.abs(image_a.astype(np.int16) - image_b.astype(np.int16))
    return diff.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 geometry sanity check for perspective/equirectangular projection."
    )
    parser.add_argument("--input", default="inputs/starry_night.jpg", help="Input perspective image.")
    parser.add_argument("--output-dir", default="outputs/stage1", help="Debug output directory.")
    parser.add_argument("--pano-width", type=int, default=4096, help="Panorama width.")
    parser.add_argument("--pano-height", type=int, default=2048, help="Panorama height.")
    parser.add_argument("--fov-x", type=float, default=70.0, help="Manual horizontal FoV in degrees.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Input yaw in degrees.")
    parser.add_argument("--pitch", type=float, default=0.0, help="Input pitch in degrees.")
    parser.add_argument("--render-width", type=int, default=0, help="Rendered front view width.")
    parser.add_argument("--render-height", type=int, default=0, help="Rendered front view height.")

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    image = load_image(args.input)
    image_height, image_width = image.shape[:2]
    render_width = args.render_width or image_width
    render_height = args.render_height or image_height

    fov_x, fov_y = estimate_or_load_fov(image, args.fov_x)
    panorama, known_mask = create_equirectangular_canvas(args.pano_width, args.pano_height)
    projected, projection_mask = project_perspective_to_equirect(
        image,
        fov_x,
        fov_y,
        args.yaw,
        args.pitch,
        (args.pano_width, args.pano_height),
    )
    panorama, known_mask = paste_projected_view(panorama, known_mask, projected, projection_mask)
    missing_mask = compute_missing_mask(known_mask)
    rendered_front = render_perspective_from_equirect(
        panorama,
        args.yaw,
        args.pitch,
        fov_x,
        fov_y,
        (render_width, render_height),
    )
    rendered_front_mask = render_perspective_mask_from_equirect(
        known_mask,
        args.yaw,
        args.pitch,
        fov_x,
        fov_y,
        (render_width, render_height),
    )

    if rendered_front.shape[:2] == image.shape[:2]:
        diff = absolute_difference(image, rendered_front)
    else:
        diff = np.zeros_like(rendered_front)

    save_image(output_dir / "00_projected_input.png", panorama)
    save_image(output_dir / "01_known_mask.png", known_mask)
    save_image(output_dir / "02_missing_mask.png", missing_mask)
    save_image(output_dir / "03_rendered_front.png", rendered_front)
    save_image(output_dir / "04_front_absdiff.png", diff)
    save_image(output_dir / "05_rendered_front_known_mask.png", rendered_front_mask)

    print("Stage 1 geometry debug artifacts written to", output_dir)
    print("fov_x:", round(fov_x, 4))
    print("fov_y:", round(fov_y, 4))


if __name__ == "__main__":
    main()
