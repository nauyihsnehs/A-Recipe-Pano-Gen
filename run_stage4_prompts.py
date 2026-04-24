import argparse

from prompt_generator import generate_panorama_prompts, save_prompt_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 4 VLM prompt generation."
    )
    parser.add_argument("--input", default="inputs/pano_aiypnfgomckrgf.jpg", help="Input perspective image.")
    parser.add_argument("--output", default="outputs/stage4/prompts.json", help="Prompt JSON output path.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--model", default="qwen3.6")
    parser.add_argument("--api-key", default="ollama")

    return parser.parse_args()


def main():
    args = parse_args()
    prompts = generate_panorama_prompts(
        args.input,
        args.base_url,
        args.model,
        args.api_key,
    )
    save_prompt_json(args.output, prompts)
    print("Stage 4 prompts written to", args.output)
    print("scene_type:", prompts["scene_type"])
    print("global_atmosphere_prompt:", prompts["global_atmosphere_prompt"])
    print("sky_or_ceiling_prompt:", prompts["sky_or_ceiling_prompt"])
    print("ground_or_floor_prompt:", prompts["ground_or_floor_prompt"])
    print("negative_prompt:", prompts["negative_prompt"])


if __name__ == "__main__":
    main()
