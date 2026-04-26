import argparse

import numpy as np

from pipeline_helper import (
    AnchoredSynthesizer,
    DebugWriter,
    DiffusersImg2ImgRefinementBackend,
    DiffusersInpaintingBackend,
    GeometryTools,
    ImageIO,
    OutputPaths,
    PanoramaRefiner,
    PromptTools,
    TomlConfigLoader,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end A Recipe panorama reproduction pipeline."
    )
    parser.add_argument("--config", default="pipeline.toml", help="Pipeline TOML config path.")

    return parser.parse_args()


def validate_config(config):
    required = [
        ("paths", ["input", "output_root"]),
        ("panorama", ["width", "height", "input_fov_x"]),
        ("view", ["size", "middle_fov", "vertical_fov"]),
        ("models", ["inpaint_model_id", "refine_model_id"]),
        ("vlm", ["base_url", "model", "api_key"]),
        ("prompting", ["mode"]),
        (
            "synthesis",
            [
                "seed",
                "num_steps",
                "guidance_scale",
                "mask_dilate_kernel",
                "mask_dilate_iterations",
                "overlap_blend",
            ],
        ),
        ("refinement", ["enabled", "steps", "guidance_scale", "denoise_strength"]),
    ]

    for section, keys in required:
        if section not in config:
            raise KeyError("missing config section: " + section)
        for key in keys:
            if key not in config[section]:
                raise KeyError("missing config key: " + section + "." + key)

    PromptTools.normalize_prompt_mode(config["prompting"]["mode"])


def resolve_prompts(config, prompt_tools, debug_dir):
    paths = config["paths"]
    vlm_config = config["vlm"]
    prompt_mode = prompt_tools.normalize_prompt_mode(config["prompting"]["mode"])

    payload = {
        "mode": prompt_mode,
    }

    if prompt_mode == "caption":
        caption_prompt = prompt_tools.generate_caption_prompt(paths["input"], vlm_config)
        prompts = prompt_tools.effective_prompts_for_mode(
            prompt_mode,
            caption_prompt=caption_prompt,
        )
        payload["caption_prompt"] = caption_prompt
    else:
        directional_prompts = prompt_tools.generate_panorama_prompts(
            paths["input"],
            vlm_config,
        )
        prompts = prompt_tools.effective_prompts_for_mode(
            prompt_mode,
            directional_prompts=directional_prompts,
        )
        payload["directional_prompts"] = directional_prompts

    payload["effective_prompts"] = prompts
    prompt_path = debug_dir / "prompts.json"
    prompt_tools.save_prompt_payload(prompt_path, payload)
    print("Generated prompts:", prompt_path, "mode:", prompt_mode)

    return prompt_tools.prompts_to_anchored_synthesis_args(prompts)


def save_anchored_synthesis_outputs(
        debug_dir,
        panorama,
        known_mask,
        input_image,
        input_fov_x,
        input_fov_y,
        pano_size,
        records,
):
    input_mask = AnchoredSynthesizer.initialize(
        input_image,
        input_fov_x,
        input_fov_y,
        pano_size,
    )[2]
    generated_mask = GeometryTools.binary_mask(known_mask)

    ImageIO.save_image(debug_dir / "80_generated_mask.png", generated_mask)
    ImageIO.save_image(debug_dir / "81_input_reference_mask.png", input_mask)
    ImageIO.save_image(debug_dir / "82_anchored_synthesis_panorama.png", panorama)
    ImageIO.save_image(debug_dir / "83_anchored_synthesis_known_mask.png", known_mask)
    ImageIO.save_image(
        debug_dir / "84_anchored_synthesis_missing_mask.png",
        GeometryTools.compute_missing_mask(known_mask),
    )
    ImageIO.save_json(debug_dir / "records.json", records)

    return generated_mask, np.zeros_like(input_mask)


def run_refinement(config, debug_dir, panorama, generated_mask, protect_mask, prompt, negative_prompt):
    refinement_config = config["refinement"]
    if not refinement_config["enabled"]:
        print("Refinement disabled")
        return panorama, []

    view_config = config["view"]
    synthesis_config = config["synthesis"]
    refine_backend = DiffusersImg2ImgRefinementBackend(config["models"]["refine_model_id"])

    def refinement_debug_writer(event, payload):
        DebugWriter.save_refinement_payload(debug_dir, event, payload)

    refined, records = PanoramaRefiner.run(
        panorama,
        generated_mask,
        protect_mask,
        prompt,
        refine_backend,
        negative_prompt=negative_prompt,
        seed=synthesis_config["seed"],
        num_steps=refinement_config["steps"],
        guidance_scale=refinement_config["guidance_scale"],
        denoise_strength=refinement_config["denoise_strength"],
        view_size=view_config["size"],
        middle_fov=view_config["middle_fov"],
        vertical_fov=view_config["vertical_fov"],
        debug_writer=refinement_debug_writer,
    )
    ImageIO.save_json(debug_dir / "refinement_records.json", records)

    return refined, records


def main():
    args = parse_args()
    config = TomlConfigLoader.load(args.config)
    validate_config(config)

    paths = config["paths"]
    panorama_config = config["panorama"]
    view_config = config["view"]
    model_config = config["models"]
    synthesis_config = config["synthesis"]
    pano_size = (panorama_config["width"], panorama_config["height"])

    run_dir, output_path, debug_dir = OutputPaths.make_run_paths(paths)
    input_image = ImageIO.load_image(paths["input"])
    input_fov_x, input_fov_y = GeometryTools.estimate_fov(input_image, panorama_config["input_fov_x"])
    global_prompt, top_prompt, bottom_prompt, negative_prompt = resolve_prompts(config, PromptTools, debug_dir)

    inpaint_backend = DiffusersInpaintingBackend(model_config["inpaint_model_id"])

    def anchored_synthesis_debug_writer(event, payload):
        DebugWriter.save_anchored_synthesis_payload(debug_dir / "anchored", event, payload)

    panorama, known_mask, anchored_synthesis_records = AnchoredSynthesizer.run(
        input_image,
        input_fov_x,
        input_fov_y,
        pano_size,
        inpaint_backend,
        global_prompt,
        top_prompt,
        bottom_prompt,
        negative_prompt,
        seed=synthesis_config["seed"],
        num_steps=synthesis_config["num_steps"],
        guidance_scale=synthesis_config["guidance_scale"],
        mask_dilate_kernel=synthesis_config["mask_dilate_kernel"],
        mask_dilate_iterations=synthesis_config["mask_dilate_iterations"],
        overlap_blend=synthesis_config["overlap_blend"],
        view_size=view_config["size"],
        middle_fov=view_config["middle_fov"],
        vertical_fov=view_config["vertical_fov"],
        debug_writer=anchored_synthesis_debug_writer,
    )
    generated_mask, refine_protect_mask = save_anchored_synthesis_outputs(
        debug_dir,
        panorama,
        known_mask,
        input_image,
        input_fov_x,
        input_fov_y,
        pano_size,
        anchored_synthesis_records,
    )

    final_panorama, refinement_records = run_refinement(
        config,
        debug_dir,
        panorama,
        generated_mask,
        refine_protect_mask,
        global_prompt,
        negative_prompt,
    )

    ImageIO.save_image(output_path, final_panorama)
    ImageIO.save_image(debug_dir / "99_final_pano.png", final_panorama)
    print("Output folder:", run_dir)
    print("Final panorama written to", output_path)
    print("Debug artifacts written to", debug_dir)
    print("anchored synthesis views:", len(anchored_synthesis_records))
    print("refinement records:", len(refinement_records))


if __name__ == "__main__":
    main()
