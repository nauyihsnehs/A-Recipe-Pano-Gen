import base64
import io
import json
import mimetypes
import re
from pathlib import Path

import numpy as np
from PIL import Image


PROMPT_FIELDS = [
    "scene_type",
    "global_atmosphere_prompt",
    "sky_or_ceiling_prompt",
    "ground_or_floor_prompt",
    "negative_prompt",
]


DEFAULT_SYSTEM_PROMPT = """
You generate panorama outpainting prompts from one input image.
Return only one JSON object. Do not include markdown fences or commentary.
The JSON object must contain exactly these string fields:
scene_type, global_atmosphere_prompt, sky_or_ceiling_prompt, ground_or_floor_prompt, negative_prompt.
scene_type must be one of indoor, outdoor, uncertain.
""".strip()


DEFAULT_USER_PROMPT = """
Given the input image, infer the scene type and generate prompts for panorama outpainting.

Return:
1. A coarse global atmosphere prompt describing the place, lighting, weather/time if visible, materials, and visual style.
   Do not describe central foreground objects, people, text, or unique objects that should not be repeated.
2. A prompt for the upper hemisphere.
   If outdoor, describe sky, clouds, lighting, tree canopy, building tops, or other plausible upper-scene elements.
   If indoor, describe ceiling, upper walls, lighting fixtures, beams, or upper architectural structures.
3. A prompt for the lower hemisphere.
   If outdoor, describe ground, road, grass, terrain, water, or floor-like surfaces.
   If indoor, describe floor material, lower walls, rugs, or lower furniture boundaries.
4. A negative prompt listing central objects, people, text, logos, duplicate foreground objects, and artifacts that should not be repeated.
""".strip()


def image_to_data_url(image):
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        data = path.read_bytes()
    else:
        buffer = io.BytesIO()
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            if isinstance(image, np.ndarray):
                array = image
            else:
                array = np.asarray(image)
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array).convert("RGB")
        pil_image.save(buffer, format="PNG")
        mime_type = "image/png"
        data = buffer.getvalue()

    encoded = base64.b64encode(data).decode("ascii")

    return "data:" + mime_type + ";base64," + encoded


def extract_json_object(text):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("VLM response did not contain a JSON object")

    return json.loads(text[start : end + 1])


def validate_prompt_schema(prompts):
    if not isinstance(prompts, dict):
        raise ValueError("prompt schema must be a JSON object")

    missing = [field for field in PROMPT_FIELDS if field not in prompts]
    if missing:
        raise ValueError("prompt schema missing fields: " + ", ".join(missing))

    normalized = {}
    for field in PROMPT_FIELDS:
        value = prompts[field]
        if value is None:
            value = ""
        if not isinstance(value, str):
            value = str(value)
        normalized[field] = value.strip()

    scene_type = normalized["scene_type"].lower()
    if scene_type not in ["indoor", "outdoor", "uncertain"]:
        scene_type = "uncertain"
    normalized["scene_type"] = scene_type

    for field in PROMPT_FIELDS:
        if field != "negative_prompt" and not normalized[field]:
            raise ValueError("prompt schema field is empty: " + field)

    return normalized


def load_prompt_json(path):
    path = Path(path)

    return validate_prompt_schema(json.loads(path.read_text(encoding="utf-8")))


def save_prompt_json(path, prompts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prompts = validate_prompt_schema(prompts)
    path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prompts_to_stage3_args(prompts):
    prompts = validate_prompt_schema(prompts)

    return (
        prompts["global_atmosphere_prompt"],
        prompts["sky_or_ceiling_prompt"],
        prompts["ground_or_floor_prompt"],
        prompts["negative_prompt"],
    )


def _completion_content(response):
    message = response.choices[0].message
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                parts.append(item.text)

        return "".join(parts)

    return str(content)


def _create_completion(client, model, messages):
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except Exception as error:
        status_code = getattr(error, "status_code", None)
        if status_code != 400 and "BadRequest" not in error.__class__.__name__:
            raise

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )


def generate_panorama_prompts(image, base_url, model, api_key):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    image_url = image_to_data_url(image)
    messages = [
        {
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": DEFAULT_USER_PROMPT,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
            ],
        },
    ]
    response = _create_completion(client, model, messages)
    content = _completion_content(response)
    prompts = extract_json_object(content)

    return validate_prompt_schema(prompts)
