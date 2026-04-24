import argparse
import copy


DEFAULT_NEGATIVE_PROMPT = "text, watermark, logo, signature, person, people, human, face, portrait, frame, border, low quality, words, letters, caption, man, woman, child, crowd, black border, white border, worst quality, chinese text, english text, subtitle, body, hands, panel, blurry, lowres, jpeg artifacts, deformed"

DEFAULT_CONFIG = {
    "paths": {
        "input": "inputs/pano_aiypnfgomckrgf.jpg",
        "output": "outputs/final/final_pano.png",
        "debug_dir": "outputs/final/debug",
        "prompt_json": "outputs/stage4/prompts.json",
        "prompt_output": "outputs/stage4/prompts.json",
    },
    "panorama": {
        "width": 2048,
        "height": 1024,
        "input_fov_x": 70.0,
    },
    "view": {
        "size": 512,
        "middle_fov": 85.0,
        "vertical_fov": 120.0,
    },
    "models": {
        "inpaint_model_id": "stabilityai/stable-diffusion-2-inpainting",
        "refine_model_id": "stabilityai/stable-diffusion-2-1-base",
        "device": "auto",
        "torch_dtype": "auto",
        "allow_download": False,
    },
    "prompts": {
        "generate": False,
        "global_prompt": "a coherent panoramic continuation of the scene",
        "top_prompt": "upper hemisphere, sky or ceiling consistent with the scene",
        "bottom_prompt": "lower hemisphere, ground or floor consistent with the scene",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    },
    "vlm": {
        "base_url": "http://127.0.0.1:11435/v1",
        "model": "qwen3.6",
        "api_key": "ollama",
    },
    "synthesis": {
        "seed": 42,
        "num_steps": 40,
        "guidance_scale": 7.5,
    },
    "refinement": {
        "enabled": True,
        "steps": 30,
        "guidance_scale": 7.5,
        "denoise_strength": 0.3,
        "mask_blur": 32.0,
    },
}


def merge_config(base, override):
    merged = copy.deepcopy(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value

    return merged


def set_config_value(config, section, key, value):
    if value is None:
        return

    config.setdefault(section, {})[key] = value


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end A Recipe panorama reproduction pipeline."
    )
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")
    parser.add_argument("--input", default=None, help="Input perspective image.")
    parser.add_argument("--output", default=None, help="Final panorama output path.")
    parser.add_argument("--debug-dir", default=None, help="Intermediate debug output directory.")
    parser.add_argument("--prompt-json", default=None, help="Load prompt JSON when present.")
    parser.add_argument("--prompt-output", default=None, help="Generated prompt JSON output path.")
    parser.add_argument("--generate-prompts", action="store_true", default=None)
    parser.add_argument("--disable-refinement", action="store_true", default=None)
    parser.add_argument("--allow-download", action="store_true", default=None)
    parser.add_argument("--pano-width", type=int, default=None, help="Panorama width.")
    parser.add_argument("--pano-height", type=int, default=None, help="Panorama height.")
    parser.add_argument("--input-fov-x", type=float, default=None, help="Input horizontal FoV.")
    parser.add_argument("--view-size", type=int, default=None, help="Square inpainting/refinement view size.")
    parser.add_argument("--middle-fov", type=float, default=None, help="Horizontal sweep view FoV.")
    parser.add_argument("--vertical-fov", type=float, default=None, help="Top and bottom view FoV.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--refine-model-id", default=None)
    parser.add_argument("--global-prompt", default=None)
    parser.add_argument("--top-prompt", default=None)
    parser.add_argument("--bottom-prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--vlm-base-url", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--vlm-api-key", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--refine-steps", type=int, default=None)
    parser.add_argument("--refine-guidance-scale", type=float, default=None)
    parser.add_argument("--denoise-strength", type=float, default=None)
    parser.add_argument("--mask-blur", type=float, default=None)
    parser.add_argument("--device", default=None, help="auto, cuda, or cpu.")
    parser.add_argument("--torch-dtype", default=None, help="auto, float16, float32, or bfloat16.")

    return parser.parse_args()


def apply_cli_overrides(config, args):
    set_config_value(config, "paths", "input", args.input)
    set_config_value(config, "paths", "output", args.output)
    set_config_value(config, "paths", "debug_dir", args.debug_dir)
    set_config_value(config, "paths", "prompt_json", args.prompt_json)
    set_config_value(config, "paths", "prompt_output", args.prompt_output)
    set_config_value(config, "panorama", "width", args.pano_width)
    set_config_value(config, "panorama", "height", args.pano_height)
    set_config_value(config, "panorama", "input_fov_x", args.input_fov_x)
    set_config_value(config, "view", "size", args.view_size)
    set_config_value(config, "view", "middle_fov", args.middle_fov)
    set_config_value(config, "view", "vertical_fov", args.vertical_fov)
    set_config_value(config, "models", "inpaint_model_id", args.model_id)
    set_config_value(config, "models", "refine_model_id", args.refine_model_id)
    set_config_value(config, "models", "device", args.device)
    set_config_value(config, "models", "torch_dtype", args.torch_dtype)
    set_config_value(config, "prompts", "global_prompt", args.global_prompt)
    set_config_value(config, "prompts", "top_prompt", args.top_prompt)
    set_config_value(config, "prompts", "bottom_prompt", args.bottom_prompt)
    set_config_value(config, "prompts", "negative_prompt", args.negative_prompt)
    set_config_value(config, "vlm", "base_url", args.vlm_base_url)
    set_config_value(config, "vlm", "model", args.vlm_model)
    set_config_value(config, "vlm", "api_key", args.vlm_api_key)
    set_config_value(config, "synthesis", "seed", args.seed)
    set_config_value(config, "synthesis", "num_steps", args.num_steps)
    set_config_value(config, "synthesis", "guidance_scale", args.guidance_scale)
    set_config_value(config, "refinement", "steps", args.refine_steps)
    set_config_value(config, "refinement", "guidance_scale", args.refine_guidance_scale)
    set_config_value(config, "refinement", "denoise_strength", args.denoise_strength)
    set_config_value(config, "refinement", "mask_blur", args.mask_blur)

    if args.generate_prompts is not None:
        config["prompts"]["generate"] = args.generate_prompts
    if args.disable_refinement is not None:
        config["refinement"]["enabled"] = not args.disable_refinement
    if args.allow_download is not None:
        config["models"]["allow_download"] = args.allow_download


def load_config(path):
    from pipeline_helper import load_toml_config

    config = merge_config(DEFAULT_CONFIG, load_toml_config(path))

    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args)

    from pipeline_helper import run_pipeline

    run_pipeline(config)


if __name__ == "__main__":
    main()
