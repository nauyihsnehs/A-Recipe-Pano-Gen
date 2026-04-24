import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from anchored_synthesis import initialize_anchored_panorama, run_anchored_synthesis
from camera import estimate_or_load_fov
from inpaint_runner import DiffusersInpaintingBackend
from mask import compute_missing_mask
from prompt_generator import (
    generate_panorama_prompts,
    load_prompt_json,
    prompts_to_stage3_args,
    save_prompt_json,
)
from refinement import DiffusersImg2ImgRefinementBackend, refine_panorama
from run_stage3_anchored import save_debug_payload as save_stage3_debug_payload


DEFAULT_NEGATIVE_PROMPT = "text, watermark, logo, signature, person, people, human, face, portrait, frame, border, low quality, words, letters, caption, man, woman, child, crowd, black border, white border, worst quality, chinese text, english text, subtitle, body, hands, panel, blurry, lowres, jpeg artifacts, deformed"


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    Image.fromarray(image).save(path)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_yaw(yaw):
    yaw = int(round(yaw)) % 360

    return str(yaw).zfill(3)


def save_refinement_debug(output_dir, event, payload):
    output_dir = Path(output_dir)

    if event == "step":
        record = payload["record"]
        prefix = (
            str(record["index"]).zfill(2)
            + "_"
            + record["phase"]
            + "_yaw_"
            + format_yaw(record["yaw"])
        )
        step_dir = output_dir / "refinement_steps"
        save_image(step_dir / (prefix + "_source_view.png"), payload["source_view"])
        save_image(step_dir / (prefix + "_refine_mask.png"), payload["refine_mask"])
        save_image(step_dir / (prefix + "_refined_view.png"), payload["refined_view"])
        save_image(step_dir / (prefix + "_blended_view.png"), payload["blended_view"])
        save_image(step_dir / (prefix + "_projected_update.png"), payload["projected_update"])
        save_image(step_dir / (prefix + "_projected_update_mask.png"), payload["projected_update_mask"])
        save_image(step_dir / (prefix + "_updated_panorama.png"), payload["updated_panorama"])
        return

    if event == "final":
        save_image(output_dir / "95_refined_panorama.png", payload["panorama"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end A Recipe panorama reproduction pipeline."
    )
    parser.add_argument("--input", default="inputs/pano_aiypnfgomckrgf.jpg", help="Input perspective image.")
    parser.add_argument("--output", default="outputs/final/final_pano.png", help="Final panorama output path.")
    parser.add_argument("--debug-dir", default="outputs/final/debug", help="Intermediate debug output directory.")
    parser.add_argument("--pano-width", type=int, default=2048, help="Panorama width.")
    parser.add_argument("--pano-height", type=int, default=1024, help="Panorama height.")
    parser.add_argument("--input-fov-x", type=float, default=70.0, help="Input horizontal FoV.")
    parser.add_argument("--view-size", type=int, default=512, help="Square inpainting/refinement view size.")
    parser.add_argument("--middle-fov", type=float, default=85.0, help="Horizontal sweep view FoV.")
    parser.add_argument("--vertical-fov", type=float, default=120.0, help="Top and bottom view FoV.")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-inpainting")
    parser.add_argument("--refine-model-id", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--global-prompt", default="a coherent panoramic continuation of the scene")
    parser.add_argument("--top-prompt", default="upper hemisphere, sky or ceiling consistent with the scene")
    parser.add_argument("--bottom-prompt", default="lower hemisphere, ground or floor consistent with the scene")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--prompt-json", default="outputs/stage4/prompts.json", help="Load Stage 4 prompt JSON when present.")
    parser.add_argument("--generate-prompts", action="store_true", help="Generate Stage 4 prompts before synthesis.")
    parser.add_argument("--prompt-output", default="outputs/stage4/prompts.json", help="Generated prompt JSON output path.")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--vlm-model", default="qwen3.6")
    parser.add_argument("--vlm-api-key", default="ollama")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--refine-steps", type=int, default=30)
    parser.add_argument("--refine-guidance-scale", type=float, default=7.5)
    parser.add_argument("--denoise-strength", type=float, default=0.3)
    parser.add_argument("--mask-blur", type=float, default=32.0)
    parser.add_argument("--disable-refinement", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--torch-dtype", default="auto", help="auto, float16, float32, or bfloat16.")
    parser.add_argument("--allow-download", action="store_true", help="Allow model downloads instead of local-files-only loading.")

    return parser.parse_args()


def resolve_prompts(args):
    if args.generate_prompts:
        prompts = generate_panorama_prompts(
            args.input,
            args.vlm_base_url,
            args.vlm_model,
            args.vlm_api_key,
        )
        save_prompt_json(args.prompt_output, prompts)
        print("Generated Stage 4 prompts:", args.prompt_output)

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
    output_path = Path(args.output)
    debug_dir = Path(args.debug_dir)
    local_files_only = not args.allow_download
    input_image = load_image(args.input)
    input_fov_x, input_fov_y = estimate_or_load_fov(input_image, args.input_fov_x)
    global_prompt, top_prompt, bottom_prompt, negative_prompt = resolve_prompts(args)

    inpaint_backend = DiffusersInpaintingBackend(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=local_files_only,
    )

    def stage3_debug_writer(event, payload):
        save_stage3_debug_payload(debug_dir / "anchored", event, payload)

    panorama, known_mask, stage3_records = run_anchored_synthesis(
        input_image,
        input_fov_x,
        input_fov_y,
        (args.pano_width, args.pano_height),
        inpaint_backend,
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
        debug_writer=stage3_debug_writer,
    )

    input_mask = initialize_anchored_panorama(
        input_image,
        input_fov_x,
        input_fov_y,
        (args.pano_width, args.pano_height),
    )[2]
    generated_mask = np.where((known_mask > 0) & (input_mask == 0), 255, 0).astype(np.uint8)
    save_image(debug_dir / "80_generated_mask.png", generated_mask)
    save_image(debug_dir / "81_input_preserve_mask.png", input_mask)
    save_image(debug_dir / "82_stage3_panorama.png", panorama)
    save_image(debug_dir / "83_stage3_known_mask.png", known_mask)
    save_image(debug_dir / "84_stage3_missing_mask.png", compute_missing_mask(known_mask))
    save_json(debug_dir / "stage3_records.json", stage3_records)

    final_panorama = panorama
    refinement_records = []

    if not args.disable_refinement:
        refine_backend = DiffusersImg2ImgRefinementBackend(
            args.refine_model_id,
            device=args.device,
            torch_dtype=args.torch_dtype,
            local_files_only=local_files_only,
        )

        def refinement_debug_writer(event, payload):
            save_refinement_debug(debug_dir, event, payload)

        final_panorama, refinement_records = refine_panorama(
            panorama,
            generated_mask,
            input_mask,
            global_prompt,
            refine_backend,
            negative_prompt=negative_prompt,
            seed=args.seed + 1000,
            num_steps=args.refine_steps,
            guidance_scale=args.refine_guidance_scale,
            denoise_strength=args.denoise_strength,
            mask_blur=args.mask_blur,
            view_size=args.view_size,
            middle_fov=args.middle_fov,
            vertical_fov=args.vertical_fov,
            debug_writer=refinement_debug_writer,
        )
        save_json(debug_dir / "refinement_records.json", refinement_records)
    else:
        print("Refinement disabled")

    save_image(output_path, final_panorama)
    save_image(debug_dir / "99_final_pano.png", final_panorama)
    print("Final panorama written to", output_path)
    print("Debug artifacts written to", debug_dir)
    print("stage3 views:", len(stage3_records))
    print("refinement records:", len(refinement_records))


if __name__ == "__main__":
    main()
