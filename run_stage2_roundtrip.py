import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from camera import estimate_or_load_fov
from canvas import create_equirectangular_canvas, paste_projected_view
from inpaint_runner import DiffusersInpaintingBackend, run_inpainting
from mask import compute_inpaint_mask, compute_missing_mask
from panorama_update import update_pano_with_view
from projection import (
    project_perspective_to_equirect,
    render_perspective_from_equirect,
    render_perspective_mask_from_equirect,
)
from view_scheduler import single_view_schedule


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    Image.fromarray(image).save(path)


def make_masked_view(view, mask):
    masked = view.copy()
    masked[mask > 0] = 0

    return masked


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 single-view inpainting round trip."
    )
    parser.add_argument("--input", default="inputs/starry_night.jpg", help="Input perspective image.")
    parser.add_argument("--output-dir", default="outputs/stage2", help="Debug output directory.")
    parser.add_argument("--pano-width", type=int, default=4096, help="Panorama width.")
    parser.add_argument("--pano-height", type=int, default=2048, help="Panorama height.")
    parser.add_argument("--input-fov-x", type=float, default=70.0, help="Input horizontal FoV.")
    parser.add_argument("--input-yaw", type=float, default=0.0, help="Input yaw in degrees.")
    parser.add_argument("--input-pitch", type=float, default=0.0, help="Input pitch in degrees.")
    parser.add_argument("--view-yaw", type=float, default=45.0, help="Target view yaw in degrees.")
    parser.add_argument("--view-pitch", type=float, default=0.0, help="Target view pitch in degrees.")
    parser.add_argument("--view-fov", type=float, default=85.0, help="Target view horizontal and vertical FoV.")
    parser.add_argument("--view-size", type=int, default=1024, help="Square inpainting view size.")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-inpainting")
    parser.add_argument("--prompt", default="a coherent panoramic continuation of the scene")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--torch-dtype", default="auto", help="auto, float16, float32, or bfloat16.")
    parser.add_argument("--local-files-only", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    input_image = load_image(args.input)
    input_fov_x, input_fov_y = estimate_or_load_fov(input_image, args.input_fov_x)
    panorama, known_mask = create_equirectangular_canvas(args.pano_width, args.pano_height)
    projected_input, projected_input_mask = project_perspective_to_equirect(
        input_image,
        input_fov_x,
        input_fov_y,
        args.input_yaw,
        args.input_pitch,
        (args.pano_width, args.pano_height),
    )
    panorama, known_mask = paste_projected_view(
        panorama,
        known_mask,
        projected_input,
        projected_input_mask,
    )

    view = single_view_schedule(
        yaw=args.view_yaw,
        pitch=args.view_pitch,
        fov_x=args.view_fov,
        fov_y=args.view_fov,
    )
    out_size = (args.view_size, args.view_size)
    rendered_view = render_perspective_from_equirect(
        panorama,
        view["yaw"],
        view["pitch"],
        view["fov_x"],
        view["fov_y"],
        out_size,
    )
    view_known_mask = render_perspective_mask_from_equirect(
        known_mask,
        view["yaw"],
        view["pitch"],
        view["fov_x"],
        view["fov_y"],
        out_size,
    )
    inpaint_mask = compute_inpaint_mask(view_known_mask)
    masked_view = make_masked_view(rendered_view, inpaint_mask)

    backend = DiffusersInpaintingBackend(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
    inpainted_view = run_inpainting(
        masked_view,
        inpaint_mask,
        args.prompt,
        args.negative_prompt,
        args.seed,
        backend,
        args.num_steps,
        args.guidance_scale,
    )
    updated_panorama, updated_known_mask, projected_update, projected_update_mask = update_pano_with_view(
        panorama,
        known_mask,
        inpainted_view,
        inpaint_mask,
        view["yaw"],
        view["pitch"],
        view["fov_x"],
        view["fov_y"],
    )

    save_image(output_dir / "00_initial_panorama.png", panorama)
    save_image(output_dir / "01_initial_known_mask.png", known_mask)
    save_image(output_dir / "02_initial_missing_mask.png", compute_missing_mask(known_mask))
    save_image(output_dir / "03_rendered_incomplete_view.png", rendered_view)
    save_image(output_dir / "04_view_known_mask.png", view_known_mask)
    save_image(output_dir / "05_inpaint_mask.png", inpaint_mask)
    save_image(output_dir / "06_masked_view.png", masked_view)
    save_image(output_dir / "07_inpainted_view.png", inpainted_view)
    save_image(output_dir / "08_projected_update.png", projected_update)
    save_image(output_dir / "09_projected_update_mask.png", projected_update_mask)
    save_image(output_dir / "10_updated_panorama.png", updated_panorama)
    save_image(output_dir / "11_updated_known_mask.png", updated_known_mask)

    before_count = int(np.count_nonzero(known_mask))
    after_count = int(np.count_nonzero(updated_known_mask))
    print("Stage 2 round-trip artifacts written to", output_dir)
    print("known pixels before:", before_count)
    print("known pixels after:", after_count)
    print("known pixels added:", after_count - before_count)


if __name__ == "__main__":
    main()
