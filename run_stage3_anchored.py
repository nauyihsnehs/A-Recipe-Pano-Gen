import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from anchored_synthesis import run_anchored_synthesis
from camera import estimate_or_load_fov
from inpaint_runner import DiffusersInpaintingBackend
from mask import compute_missing_mask
from prompt_generator import (
    generate_panorama_prompts,
    load_prompt_json,
    prompts_to_stage3_args,
    save_prompt_json,
)


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    Image.fromarray(image).save(path)


def format_yaw(yaw):
    yaw = int(round(yaw)) % 360

    return str(yaw).zfill(3)


def save_debug_payload(output_dir, event, payload):
    output_dir = Path(output_dir)

    if event == "initial":
        save_image(output_dir / "00_initial_front_back_anchor.png", payload["panorama"])
        save_image(output_dir / "01_initial_known_mask.png", payload["known_mask"])
        save_image(output_dir / "02_front_input_mask.png", payload["input_mask"])
        save_image(output_dir / "03_back_anchor_mask.png", payload["anchor_mask"])
        save_image(output_dir / "04_initial_missing_mask.png", compute_missing_mask(payload["known_mask"]))
        return

    if event == "after_vertical":
        save_image(output_dir / "30_after_top_bottom_panorama.png", payload["panorama"])
        save_image(output_dir / "31_after_top_bottom_known_mask.png", payload["known_mask"])
        return

    if event == "anchor_removed":
        save_image(output_dir / "40_anchor_removed_panorama.png", payload["panorama"])
        save_image(output_dir / "41_anchor_removed_known_mask.png", payload["known_mask"])
        save_image(output_dir / "42_anchor_removed_missing_mask.png", compute_missing_mask(payload["known_mask"]))
        return

    if event == "step":
        record = payload["record"]
        prefix = (
            str(record["index"]).zfill(2)
            + "_"
            + record["phase"]
            + "_yaw_"
            + format_yaw(record["yaw"])
        )
        step_dir = output_dir / "steps"
        save_image(step_dir / (prefix + "_rendered_view.png"), payload["rendered_view"])
        save_image(step_dir / (prefix + "_view_known_mask.png"), payload["view_known_mask"])
        save_image(step_dir / (prefix + "_inpaint_mask.png"), payload["inpaint_mask"])
        save_image(step_dir / (prefix + "_masked_view.png"), payload["masked_view"])
        save_image(step_dir / (prefix + "_inpainted_view.png"), payload["inpainted_view"])
        save_image(step_dir / (prefix + "_projected_update.png"), payload["projected_update"])
        save_image(step_dir / (prefix + "_projected_update_mask.png"), payload["projected_update_mask"])
        save_image(step_dir / (prefix + "_updated_panorama.png"), payload["updated_panorama"])
        save_image(step_dir / (prefix + "_updated_known_mask.png"), payload["updated_known_mask"])
        return

    if event == "final":
        save_image(output_dir / "90_final_panorama.png", payload["panorama"])
        save_image(output_dir / "91_final_known_mask.png", payload["known_mask"])
        save_image(output_dir / "92_final_missing_mask.png", compute_missing_mask(payload["known_mask"]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3 anchored panorama synthesis."
    )
    # parser.add_argument("--input", default="inputs/starry_night.jpg", help="Input perspective image.")
    parser.add_argument("--input", default="inputs/pano_aiypnfgomckrgf.jpg", help="Input perspective image.")
    parser.add_argument("--output-dir", default="outputs/stage3", help="Debug output directory.")
    # parser.add_argument("--pano-width", type=int, default=4096, help="Panorama width.")
    # parser.add_argument("--pano-height", type=int, default=2048, help="Panorama height.")
    parser.add_argument("--pano-width", type=int, default=2048, help="Panorama width.")
    parser.add_argument("--pano-height", type=int, default=1024, help="Panorama height.")
    parser.add_argument("--input-fov-x", type=float, default=70.0, help="Input horizontal FoV.")
    # parser.add_argument("--view-size", type=int, default=1024, help="Square inpainting view size.")
    parser.add_argument("--view-size", type=int, default=512, help="Square inpainting view size.")
    parser.add_argument("--middle-fov", type=float, default=85.0, help="Horizontal sweep view FoV.")
    parser.add_argument("--vertical-fov", type=float, default=120.0, help="Top and bottom view FoV.")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-inpainting")
    # parser.add_argument("--model-id", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    # parser.add_argument("--model-id", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--global-prompt", default="a coherent panoramic continuation of the scene")
    parser.add_argument("--top-prompt", default="upper hemisphere, sky or ceiling consistent with the scene")
    parser.add_argument("--bottom-prompt", default="lower hemisphere, ground or floor consistent with the scene")
    parser.add_argument("--negative-prompt", default="text, watermark, logo, signature, person, people, human, face, portrait, frame, border, low quality, words, letters, caption, man, woman, child, crowd, black border, white border, worst quality, chinese text, english text, subtitle, body, hands, panel, blurry, lowres, jpeg artifacts, deformed")
    parser.add_argument("--prompt-json", default="outputs/stage4/prompts.json", help="Load Stage 4 prompt JSON.")
    parser.add_argument("--generate-prompts", action="store_true", help="Generate Stage 4 prompts before synthesis.")
    parser.add_argument("--prompt-output", default="outputs/stage4/prompts.json", help="Generated prompt JSON output path.")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--vlm-model", default="qwen3.6")
    parser.add_argument("--vlm-api-key", default="ollama")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--torch-dtype", default="auto", help="auto, float16, float32, or bfloat16.")
    parser.add_argument("--local-files-only", action="store_true", default=True)

    return parser.parse_args()


def resolve_prompts(args):
    if args.generate_prompts:
        prompts = generate_panorama_prompts(
            args.input,
            args.vlm_base_url,
            args.vlm_model,
            args.vlm_api_key,
        )
        output_path = args.prompt_json or args.prompt_output
        save_prompt_json(output_path, prompts)
        print("Generated Stage 4 prompts:", output_path)

        return prompts_to_stage3_args(prompts)

    if args.prompt_json and Path(args.prompt_json).exists():
        prompts = load_prompt_json(args.prompt_json)
        print("Loaded Stage 4 prompts:", args.prompt_json)

        return prompts_to_stage3_args(prompts)

    if args.prompt_json:
        print("Prompt JSON not found, using CLI prompts:", args.prompt_json)

    return args.global_prompt, args.top_prompt, args.bottom_prompt, args.negative_prompt


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    input_image = load_image(args.input)
    input_fov_x, input_fov_y = estimate_or_load_fov(input_image, args.input_fov_x)
    global_prompt, top_prompt, bottom_prompt, negative_prompt = resolve_prompts(args)
    backend = DiffusersInpaintingBackend(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )

    def debug_writer(event, payload):
        save_debug_payload(output_dir, event, payload)

    final_panorama, final_known_mask, records = run_anchored_synthesis(
        input_image,
        input_fov_x,
        input_fov_y,
        (args.pano_width, args.pano_height),
        backend,
        global_prompt,
        top_prompt,
        bottom_prompt,
        negative_prompt,
        seed=args.seed,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        view_size=args.view_size,
        middle_fov=args.middle_fov,
        vertical_fov=args.vertical_fov,
        debug_writer=debug_writer,
    )

    print("Stage 3 anchored synthesis artifacts written to", output_dir)
    print("scheduled views:", len(records))
    print("known pixels final:", int(np.count_nonzero(final_known_mask)))
    print("final panorama shape:", final_panorama.shape)


if __name__ == "__main__":
    main()
