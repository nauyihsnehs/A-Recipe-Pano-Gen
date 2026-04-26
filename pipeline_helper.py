import base64
import io
import json
import math
import mimetypes
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
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

DEFAULT_CAPTION_SYSTEM_PROMPT = """
You write plain image captions for panorama prompt ablation.
Return only one concise caption. Do not include JSON, markdown fences, or commentary.
""".strip()

DEFAULT_CAPTION_USER_PROMPT = """
Describe the input image in one concise caption.
Include visible scene content, objects, lighting, and style.
""".strip()

PROMPT_MODES = ["directional", "coarse", "caption"]


class TomlConfigLoader:
    @staticmethod
    def load(path):
        try:
            import tomllib as toml_reader
        except ModuleNotFoundError:
            import tomli as toml_reader

        with Path(path).open("rb") as handle:
            return toml_reader.load(handle)


class ImageIO:
    @staticmethod
    def load_image(path):
        return np.asarray(Image.open(path).convert("RGB"))

    @staticmethod
    def to_uint8(image):
        return np.clip(image, 0, 255).astype(np.uint8) if image.dtype != np.uint8 else image

    @staticmethod
    def save_image(path, image):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ImageIO.to_uint8(image)).save(path)

    @staticmethod
    def save_json(path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class OutputPaths:
    @staticmethod
    def make_run_paths(paths):
        output_root = Path(paths["output_root"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / timestamp
        index = 1

        while run_dir.exists():
            run_dir = output_root / (timestamp + "_" + str(index).zfill(2))
            index += 1

        input_stem = Path(paths["input"]).stem
        output_path = run_dir / (input_stem + "_pano.png")
        debug_dir = run_dir / "debug"

        return run_dir, output_path, debug_dir


class GeometryTools:
    @staticmethod
    def estimate_fov(image, fov_x):
        if fov_x is None:
            raise ValueError("input_fov_x must be configured")

        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("image width and height must be positive")
        fov_x = float(fov_x)

        fov_x_rad = math.radians(fov_x)
        focal = (width * 0.5) / math.tan(fov_x_rad * 0.5)
        fov_y = math.degrees(2.0 * math.atan((height * 0.5) / focal))

        return GeometryTools._check_fov(fov_x, fov_y)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def compute_missing_mask(known_mask):
        return np.where(known_mask > 0, 0, 255).astype(np.uint8)

    @staticmethod
    def binary_mask(mask):
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    @staticmethod
    def dilate_mask(mask, kernel_size, iterations):
        mask = GeometryTools.binary_mask(mask)
        kernel_size = int(kernel_size)
        iterations = int(iterations)

        if kernel_size <= 0 or iterations <= 0:
            return mask

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        return cv2.dilate(mask, kernel, iterations=iterations)

    @staticmethod
    def _rotation_matrix(yaw, pitch):
        yaw = math.radians(float(yaw))
        pitch = math.radians(float(pitch))

        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)

        yaw_matrix = np.array(
            [
                [cy, 0.0, sy],
                [0.0, 1.0, 0.0],
                [-sy, 0.0, cy],
            ],
            dtype=np.float32,
        )
        pitch_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cp, sp],
                [0.0, -sp, cp],
            ],
            dtype=np.float32,
        )

        return yaw_matrix @ pitch_matrix

    @staticmethod
    def _check_fov(fov_x, fov_y):
        fov_x = float(fov_x)
        fov_y = float(fov_y)

        if fov_x <= 0.0 or fov_x >= 180.0:
            raise ValueError("fov_x must be between 0 and 180 degrees")
        if fov_y <= 0.0 or fov_y >= 180.0:
            raise ValueError("fov_y must be between 0 and 180 degrees")

        return fov_x, fov_y

    @staticmethod
    def _parse_size(size):
        if len(size) != 2:
            raise ValueError("size must be a (width, height) pair")

        width = int(size[0])
        height = int(size[1])

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")

        return width, height

    @staticmethod
    def _perspective_rays(width, height, fov_x, fov_y):
        fov_x, fov_y = GeometryTools._check_fov(fov_x, fov_y)
        tan_x = math.tan(math.radians(fov_x) * 0.5)
        tan_y = math.tan(math.radians(fov_y) * 0.5)

        x = (np.arange(width, dtype=np.float32) + 0.5) / width
        y = (np.arange(height, dtype=np.float32) + 0.5) / height
        x = (x * 2.0 - 1.0) * tan_x
        y = (y * 2.0 - 1.0) * tan_y

        xv, yv = np.meshgrid(x, y)
        rays = np.stack([xv, -yv, np.ones_like(xv)], axis=-1)
        norm = np.linalg.norm(rays, axis=-1, keepdims=True)

        return rays / norm

    @staticmethod
    def _equirectangular_rays(width, height):
        x = (np.arange(width, dtype=np.float32) + 0.5) / width
        y = (np.arange(height, dtype=np.float32) + 0.5) / height

        theta = x * (2.0 * math.pi) - math.pi
        latitude = (math.pi * 0.5) - y * math.pi

        theta, latitude = np.meshgrid(theta, latitude)
        cos_latitude = np.cos(latitude)

        rays = np.stack(
            [
                cos_latitude * np.sin(theta),
                np.sin(latitude),
                cos_latitude * np.cos(theta),
            ],
            axis=-1,
        )

        return rays.astype(np.float32)

    @staticmethod
    def _compute_projection_maps(pano_width, pano_height, src_width, src_height, fov_x, fov_y, yaw, pitch):
        fov_x, fov_y = GeometryTools._check_fov(fov_x, fov_y)
        tan_x = math.tan(math.radians(fov_x) * 0.5)
        tan_y = math.tan(math.radians(fov_y) * 0.5)
        rotation = GeometryTools._rotation_matrix(yaw, pitch)
        world_rays = GeometryTools._equirectangular_rays(pano_width, pano_height)
        camera_rays = world_rays @ rotation
        camera_z = camera_rays[..., 2]
        x_norm = np.zeros_like(camera_z, dtype=np.float32)
        y_norm = np.zeros_like(camera_z, dtype=np.float32)
        visible = camera_z > 1e-6
        np.divide(camera_rays[..., 0], camera_z, out=x_norm, where=visible)
        np.divide(camera_rays[..., 1], camera_z, out=y_norm, where=visible)
        map_x = ((x_norm / tan_x) + 1.0) * (src_width * 0.5) - 0.5
        map_y = ((-y_norm / tan_y) + 1.0) * (src_height * 0.5) - 0.5
        valid = (
                visible
                & (np.abs(x_norm) <= tan_x)
                & (np.abs(y_norm) <= tan_y)
                & (map_x >= 0.0)
                & (map_x <= src_width - 1)
                & (map_y >= 0.0)
                & (map_y <= src_height - 1)
        )
        return map_x, map_y, valid

    @staticmethod
    def _project_perspective(data, fov_x, fov_y, yaw, pitch, pano_size, interpolation):
        pano_width, pano_height = GeometryTools._parse_size(pano_size)
        src_height, src_width = data.shape[:2]

        if pano_width != pano_height * 2:
            raise ValueError("pano_size must use a 2:1 width:height ratio")

        map_x, map_y, valid = GeometryTools._compute_projection_maps(
            pano_width, pano_height, src_width, src_height, fov_x, fov_y, yaw, pitch
        )
        map_x = np.where(valid, map_x, 0.0).astype(np.float32)
        map_y = np.where(valid, map_y, 0.0).astype(np.float32)

        projected = cv2.remap(
            data,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        footprint = np.where(valid, 255, 0).astype(np.uint8)
        projected[footprint == 0] = 0

        return projected, footprint

    @staticmethod
    def project_perspective_to_equirect(image, fov_x, fov_y, yaw, pitch, pano_size):
        return GeometryTools._project_perspective(
            image,
            fov_x,
            fov_y,
            yaw,
            pitch,
            pano_size,
            cv2.INTER_LINEAR,
        )

    @staticmethod
    def project_perspective_mask_to_equirect(mask, fov_x, fov_y, yaw, pitch, pano_size):
        projected, footprint = GeometryTools._project_perspective(
            mask,
            fov_x,
            fov_y,
            yaw,
            pitch,
            pano_size,
            cv2.INTER_NEAREST,
        )

        return GeometryTools.binary_mask(np.where(footprint > 0, projected, 0))

    @staticmethod
    def _render_perspective_impl(data, yaw, pitch, fov_x, fov_y, view_render_size, interpolation):
        out_width, out_height = GeometryTools._parse_size(view_render_size)
        pano_height, pano_width = data.shape[:2]
        fov_x, fov_y = GeometryTools._check_fov(fov_x, fov_y)
        rays = GeometryTools._perspective_rays(out_width, out_height, fov_x, fov_y)
        rotation = GeometryTools._rotation_matrix(yaw, pitch)
        world_rays = rays @ rotation.T
        longitude = np.arctan2(world_rays[..., 0], world_rays[..., 2])
        latitude = np.arcsin(np.clip(world_rays[..., 1], -1.0, 1.0))
        map_x = ((longitude + math.pi) / (2.0 * math.pi)) * pano_width - 0.5
        map_y = ((math.pi * 0.5 - latitude) / math.pi) * pano_height - 0.5
        map_x = np.mod(map_x, pano_width).astype(np.float32)
        map_y = np.clip(map_y, 0.0, pano_height - 1).astype(np.float32)
        return cv2.remap(
            data,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_WRAP,
        )

    @staticmethod
    def render_perspective_from_equirect(panorama, yaw, pitch, fov_x, fov_y, view_render_size):
        return GeometryTools._render_perspective_impl(
            panorama,
            yaw,
            pitch,
            fov_x,
            fov_y,
            view_render_size,
            cv2.INTER_LINEAR,
        )

    @staticmethod
    def render_perspective_mask_from_equirect(mask, yaw, pitch, fov_x, fov_y, view_render_size):
        return GeometryTools._render_perspective_impl(
            mask,
            yaw,
            pitch,
            fov_x,
            fov_y,
            view_render_size,
            cv2.INTER_NEAREST,
        )


class ViewSchedule:
    @staticmethod
    def anchored(middle_fov, vertical_fov):
        views = []
        vertical_fov = float(vertical_fov)
        vertical_pitch = 45.0

        for phase, yaw, pitch in [
            ("top", 0, vertical_pitch),
            ("top", 180, vertical_pitch),
            ("bottom", 0, -vertical_pitch),
            ("bottom", 180, -vertical_pitch),
        ]:
            views.append(
                SimpleNamespace(
                    stage="front_back_vertical",
                    phase=phase,
                    yaw=float(yaw),
                    pitch=pitch,
                    fov_x=vertical_fov,
                    fov_y=vertical_fov,
                )
            )

        for yaw in np.arange(22.5, 360.0, 45.0):
            views.append(
                SimpleNamespace(
                    stage="horizontal",
                    phase="horizontal",
                    yaw=float(yaw),
                    pitch=0.0,
                    fov_x=float(middle_fov),
                    fov_y=float(middle_fov),
                )
            )

        return views


class DiffusersLoader:
    @staticmethod
    def cuda_device():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this pipeline")

        return "cuda"

    @staticmethod
    def float16_dtype():
        import torch

        return torch.float16

    @staticmethod
    def from_pretrained(pipeline_class, model_id, kwargs):
        kwargs = kwargs.copy()
        kwargs["variant"] = "fp16"

        try:
            return pipeline_class.from_pretrained(model_id, **kwargs)
        except Exception:
            kwargs.pop("variant", None)

        return pipeline_class.from_pretrained(model_id, **kwargs)


class DiffusersBackendBase:
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = "cuda"
        self.pipeline = None

    def load(self):
        device = DiffusersLoader.cuda_device()
        torch_dtype = DiffusersLoader.float16_dtype()
        self.pipeline = DiffusersLoader.from_pretrained(
            self.pipeline_class,
            self.model_id,
            {"torch_dtype": torch_dtype, "safety_checker": None},
        )
        self.pipeline = self.pipeline.to(device)
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
        self.device = device
        return self.pipeline

    def make_generator(self, seed):
        import torch
        return torch.Generator(device=self.device).manual_seed(int(seed))


class DiffusersInpaintingBackend(DiffusersBackendBase):
    def __init__(self, model_id):
        from diffusers import DiffusionPipeline
        super().__init__(model_id)
        self.pipeline_class = DiffusionPipeline

    def __call__(
            self,
            image,
            mask,
            prompt,
            negative_prompt=None,
            seed=42,
            generator=None,
            num_steps=40,
            guidance_scale=7.5,
    ):
        import torch

        if self.pipeline is None:
            self.load()

        image = ImageIO.to_uint8(image)
        mask = GeometryTools.binary_mask(mask)
        image_pil = Image.fromarray(image).convert("RGB")
        mask_pil = Image.fromarray(mask).convert("L")

        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            height=image.shape[0],
            width=image.shape[1],
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )

        return np.asarray(result.images[0].convert("RGB"))


class DiffusersImg2ImgRefinementBackend(DiffusersBackendBase):
    def __init__(self, model_id):
        from diffusers import AutoPipelineForImage2Image
        super().__init__(model_id)
        self.pipeline_class = AutoPipelineForImage2Image

    def __call__(
            self,
            image,
            prompt,
            negative_prompt=None,
            seed=42,
            generator=None,
            num_steps=30,
            guidance_scale=7.5,
            denoise_strength=0.3,
    ):
        import torch

        if self.pipeline is None:
            self.load()

        image = ImageIO.to_uint8(image)
        image_pil = Image.fromarray(image).convert("RGB")

        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            height=image.shape[0],
            width=image.shape[1],
            strength=float(denoise_strength),
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )

        return np.asarray(result.images[0].convert("RGB"))


class PanoramaUpdater:
    @staticmethod
    def update_with_view(
            panorama,
            known_mask,
            generated_view,
            update_mask,
            yaw,
            pitch,
            fov_x,
            fov_y,
            protect_mask=None,
            overlap_blend=False,
    ):
        if panorama.shape[:2] != known_mask.shape:
            raise ValueError("panorama and known_mask dimensions do not match")
        if generated_view.shape[:2] != update_mask.shape:
            raise ValueError("generated_view and update_mask dimensions do not match")

        pano_size = (panorama.shape[1], panorama.shape[0])
        projected_view, view_footprint = GeometryTools.project_perspective_to_equirect(
            generated_view,
            fov_x,
            fov_y,
            yaw,
            pitch,
            pano_size,
        )
        projected_mask = GeometryTools.project_perspective_mask_to_equirect(
            update_mask,
            fov_x,
            fov_y,
            yaw,
            pitch,
            pano_size,
        )
        if protect_mask is None:
            protect_mask = np.zeros_like(known_mask)
        else:
            protect_mask = GeometryTools.binary_mask(protect_mask)

        projected_update_mask = np.where(
            (projected_mask > 0)
            & (view_footprint > 0)
            & (known_mask == 0)
            & (protect_mask == 0),
            255,
            0,
        ).astype(np.uint8)

        region = projected_update_mask > 0
        projected_update = np.zeros_like(panorama)
        projected_update[region] = projected_view[region]

        updated_panorama = panorama.copy()
        updated_known_mask = known_mask.copy()
        updated_panorama[region] = projected_view[region]
        updated_known_mask[region] = 255

        projected_blend_mask = np.zeros_like(known_mask)
        if overlap_blend:
            projected_blend_mask = np.where(
                (projected_mask > 0)
                & (view_footprint > 0)
                & (known_mask > 0)
                & (protect_mask == 0),
                projected_mask,
                0,
            ).astype(np.uint8)
            blend_region = projected_blend_mask > 0

            if np.any(blend_region):
                alpha = projected_blend_mask.astype(np.float32) / 255.0
                alpha = alpha[..., None]
                blended = (
                        projected_view.astype(np.float32) * alpha
                        + updated_panorama.astype(np.float32) * (1.0 - alpha)
                )
                updated_panorama[blend_region] = np.clip(blended[blend_region], 0, 255).astype(np.uint8)

        return updated_panorama, updated_known_mask, projected_update, projected_update_mask, projected_blend_mask


class AnchoredSynthesizer:
    @staticmethod
    def initialize(input_image, input_fov_x, input_fov_y, pano_size):
        panorama, known_mask = GeometryTools.create_equirectangular_canvas(pano_size[0], pano_size[1])
        front_image, input_mask = GeometryTools.project_perspective_to_equirect(
            input_image,
            input_fov_x,
            input_fov_y,
            0.0,
            0.0,
            pano_size,
        )
        panorama, known_mask = GeometryTools.paste_projected_view(
            panorama,
            known_mask,
            front_image,
            input_mask,
        )

        anchor_image, anchor_mask = GeometryTools.project_perspective_to_equirect(
            input_image,
            input_fov_x,
            input_fov_y,
            180.0,
            0.0,
            pano_size,
        )
        panorama, known_mask = GeometryTools.paste_projected_view(
            panorama,
            known_mask,
            anchor_image,
            anchor_mask,
        )

        input_mask = GeometryTools.binary_mask(input_mask)
        anchor_mask = GeometryTools.binary_mask(anchor_mask)

        return panorama, known_mask, input_mask, anchor_mask

    @staticmethod
    def remove_backside_anchor(panorama, known_mask, input_mask, anchor_mask):
        output = panorama.copy()
        output_mask = known_mask.copy()
        anchor_only = (anchor_mask > 0) & (input_mask == 0)

        output[anchor_only] = 0
        output_mask[anchor_only] = 0
        output_mask[input_mask > 0] = 255

        return output, output_mask

    @staticmethod
    def make_masked_view(view, mask):
        alpha = np.clip(mask, 0, 255).astype(np.float32) / 255.0
        masked = view.astype(np.float32) * (1.0 - alpha[..., None])

        return np.clip(masked, 0, 255).astype(np.uint8)

    @staticmethod
    def combine_prompts(primary, secondary):
        parts = [p.strip() for p in (primary, secondary) if p.strip()]
        return ", ".join(parts)

    @staticmethod
    def prompt_for_view(view, global_prompt, top_prompt, bottom_prompt):
        if view.phase == "top":
            return AnchoredSynthesizer.combine_prompts(top_prompt, global_prompt)
        if view.phase == "bottom":
            return AnchoredSynthesizer.combine_prompts(bottom_prompt, global_prompt)

        return global_prompt

    @staticmethod
    def run_step(
            panorama,
            known_mask,
            view,
            backend,
            prompt,
            negative_prompt,
            seed,
            generator,
            num_steps,
            guidance_scale,
            view_size,
            mask_dilate_kernel,
            mask_dilate_iterations,
            protect_mask,
            overlap_blend,
    ):
        view_render_size = (view_size, view_size)
        rendered_view = GeometryTools.render_perspective_from_equirect(
            panorama, view.yaw, view.pitch, view.fov_x, view.fov_y, view_render_size,
        )
        view_known_mask = GeometryTools.render_perspective_mask_from_equirect(
            known_mask, view.yaw, view.pitch, view.fov_x, view.fov_y, view_render_size,
        )
        raw_inpaint_mask = GeometryTools.compute_missing_mask(view_known_mask)
        inpaint_mask = GeometryTools.dilate_mask(
            raw_inpaint_mask, mask_dilate_kernel, mask_dilate_iterations,
        )
        masked_view = AnchoredSynthesizer.make_masked_view(rendered_view, inpaint_mask)
        inpainted_view = backend(
            masked_view,
            inpaint_mask,
            prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            generator=generator,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        (
            updated_panorama,
            updated_known_mask,
            projected_update,
            projected_update_mask,
            projected_blend_mask,
        ) = PanoramaUpdater.update_with_view(
            panorama, known_mask, inpainted_view, inpaint_mask,
            view.yaw, view.pitch, view.fov_x, view.fov_y,
            protect_mask=protect_mask,
            overlap_blend=overlap_blend,
        )

        debug_images = {
            "rendered_view": rendered_view,
            "view_known_mask": view_known_mask,
            "raw_inpaint_mask": raw_inpaint_mask,
            "inpaint_mask": inpaint_mask,
            "masked_view": masked_view,
            "inpainted_view": inpainted_view,
            "projected_update": projected_update,
            "projected_update_mask": projected_update_mask,
            "projected_blend_mask": projected_blend_mask,
            "updated_panorama": updated_panorama,
            "updated_known_mask": updated_known_mask,
        }

        return updated_panorama, updated_known_mask, debug_images

    @staticmethod
    def run(
            input_image,
            input_fov_x,
            input_fov_y,
            pano_size,
            backend,
            global_prompt,
            top_prompt,
            bottom_prompt,
            negative_prompt,
            seed,
            num_steps,
            guidance_scale,
            mask_dilate_kernel,
            mask_dilate_iterations,
            overlap_blend,
            view_size,
            middle_fov,
            vertical_fov,
            debug_writer=None,
    ):
        panorama, known_mask, input_mask, anchor_mask = AnchoredSynthesizer.initialize(
            input_image,
            input_fov_x,
            input_fov_y,
            pano_size,
        )
        stitched_panorama, stitched_known_mask = GeometryTools.create_equirectangular_canvas(
            pano_size[0],
            pano_size[1],
        )
        records = []
        anchor_removed = False
        horizontal_saved = False
        schedule = ViewSchedule.anchored(middle_fov=middle_fov, vertical_fov=vertical_fov)
        generator = None
        if hasattr(backend, "make_generator"):
            generator = backend.make_generator(seed)

        if debug_writer:
            debug_writer(
                "initial",
                {
                    "panorama": panorama,
                    "known_mask": known_mask,
                    "input_mask": input_mask,
                    "anchor_mask": anchor_mask,
                },
            )

        for index, view in enumerate(schedule):
            if view.stage == "horizontal" and not anchor_removed:
                panorama, known_mask = AnchoredSynthesizer.remove_backside_anchor(
                    panorama,
                    known_mask,
                    input_mask,
                    anchor_mask,
                )
                anchor_removed = True

                if debug_writer:
                    debug_writer(
                        "after_front_back_vertical",
                        {
                            "panorama": panorama,
                            "known_mask": known_mask,
                        },
                    )

            known_before = int(np.count_nonzero(known_mask))
            stitch_before = int(np.count_nonzero(stitched_known_mask))
            prompt = AnchoredSynthesizer.prompt_for_view(view, global_prompt, top_prompt, bottom_prompt)
            protect_mask = input_mask
            if not anchor_removed:
                protect_mask = np.maximum(input_mask, anchor_mask)
            panorama, known_mask, debug_images = AnchoredSynthesizer.run_step(
                panorama,
                known_mask,
                view,
                backend,
                prompt,
                negative_prompt,
                int(seed),
                generator,
                num_steps,
                guidance_scale,
                view_size,
                mask_dilate_kernel,
                mask_dilate_iterations,
                protect_mask,
                overlap_blend,
            )
            stitch_mask = np.full(
                debug_images["inpaint_mask"].shape,
                255,
                dtype=np.uint8,
            )
            (
                stitched_panorama,
                stitched_known_mask,
                stitched_view_update,
                stitched_view_update_mask,
                stitched_view_blend_mask,
            ) = PanoramaUpdater.update_with_view(
                stitched_panorama,
                stitched_known_mask,
                debug_images["inpainted_view"],
                stitch_mask,
                view.yaw,
                view.pitch,
                view.fov_x,
                view.fov_y,
                overlap_blend=overlap_blend,
            )
            known_after = int(np.count_nonzero(known_mask))
            stitch_after = int(np.count_nonzero(stitched_known_mask))
            record = {
                "index": index,
                "stage": view.stage,
                "phase": view.phase,
                "yaw": view.yaw,
                "pitch": view.pitch,
                "fov_x": view.fov_x,
                "fov_y": view.fov_y,
                "known_before": known_before,
                "known_after": known_after,
                "known_added": known_after - known_before,
                "stitch_before": stitch_before,
                "stitch_after": stitch_after,
                "stitch_added": stitch_after - stitch_before,
                "prompt": prompt,
            }
            records.append(record)

            debug_images.update(
                stitched_view_update=stitched_view_update,
                stitched_view_update_mask=stitched_view_update_mask,
                stitched_view_blend_mask=stitched_view_blend_mask,
                stitched_panorama=stitched_panorama,
                stitched_known_mask=stitched_known_mask,
            )
            payload = debug_images.copy()
            payload["record"] = record
            if debug_writer:
                debug_writer("step", payload)

            last_horizontal_view = (
                    view.stage == "horizontal"
                    and (index + 1 >= len(schedule) or schedule[index + 1].stage != "horizontal")
            )
            if last_horizontal_view and not horizontal_saved:
                horizontal_saved = True
                if debug_writer:
                    debug_writer(
                        "after_horizontal",
                        {
                            "panorama": panorama,
                            "known_mask": known_mask,
                            "stitched_panorama": stitched_panorama,
                            "stitched_known_mask": stitched_known_mask,
                        },
                    )

        if debug_writer:
            debug_writer(
                "final",
                {
                    "panorama": stitched_panorama,
                    "known_mask": stitched_known_mask,
                    "context_panorama": panorama,
                    "context_known_mask": known_mask,
                },
            )

        return stitched_panorama, stitched_known_mask, records


class PanoramaRefiner:
    @staticmethod
    def project_refined_view(panorama, refined_view, refine_mask, view, protect_mask):
        pano_size = (panorama.shape[1], panorama.shape[0])
        projected_view, footprint = GeometryTools.project_perspective_to_equirect(refined_view, view.fov_x, view.fov_y, view.yaw, view.pitch, pano_size)
        projected_mask = GeometryTools.project_perspective_mask_to_equirect(refine_mask, view.fov_x, view.fov_y, view.yaw, view.pitch, pano_size)
        projected_update_mask = np.where((projected_mask > 0) & (footprint > 0) & (protect_mask == 0), 255, 0).astype(np.uint8)

        output = panorama.copy()
        projected_update = np.zeros_like(panorama)
        region = projected_update_mask > 0
        output[region] = projected_view[region]
        projected_update[region] = projected_view[region]

        return output, projected_update, projected_update_mask

    @staticmethod
    def run(
            panorama,
            generated_mask,
            protect_mask,
            prompt,
            backend,
            negative_prompt,
            seed,
            num_steps,
            guidance_scale,
            denoise_strength,
            view_size,
            middle_fov,
            vertical_fov,
            debug_writer=None,
    ):
        refined = panorama.copy()
        records = []
        schedule = ViewSchedule.anchored(middle_fov=middle_fov, vertical_fov=vertical_fov)
        generator_seed = int(seed) + 1000
        generator = None
        if hasattr(backend, "make_generator"):
            generator = backend.make_generator(generator_seed)

        for index, view in enumerate(schedule):
            view_render_size = (view_size, view_size)
            view_generated_mask = GeometryTools.render_perspective_mask_from_equirect(
                generated_mask,
                view.yaw,
                view.pitch,
                view.fov_x,
                view.fov_y,
                view_render_size,
            )
            view_protect_mask = GeometryTools.render_perspective_mask_from_equirect(
                protect_mask,
                view.yaw,
                view.pitch,
                view.fov_x,
                view.fov_y,
                view_render_size,
            )
            refine_mask = np.where((view_generated_mask > 0) & (view_protect_mask == 0), 255, 0).astype(np.uint8)
            pixel_count = int(np.count_nonzero(refine_mask))
            record = {
                "index": index,
                "stage": view.stage,
                "phase": view.phase,
                "yaw": view.yaw,
                "pitch": view.pitch,
                "refine_pixels": pixel_count,
            }

            if pixel_count == 0:
                record["skipped"] = True
                records.append(record)
                continue

            source_view = GeometryTools.render_perspective_from_equirect(
                refined,
                view.yaw,
                view.pitch,
                view.fov_x,
                view.fov_y,
                view_render_size,
            )
            refined_view = backend(
                source_view,
                prompt,
                negative_prompt=negative_prompt,
                seed=generator_seed,
                generator=generator,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                denoise_strength=denoise_strength,
            )
            alpha = np.where(refine_mask > 0, 1.0, 0.0).astype(np.float32)
            alpha = alpha[..., None]
            blended_view = (refined_view.astype(np.float32) * alpha + source_view.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            refined, projected_update, projected_update_mask = PanoramaRefiner.project_refined_view(
                refined,
                blended_view,
                refine_mask,
                view,
                protect_mask,
            )
            record["skipped"] = False
            record["projected_pixels"] = int(np.count_nonzero(projected_update_mask))
            records.append(record)

            if debug_writer:
                debug_writer(
                    "step",
                    {
                        "record": record,
                        "source_view": source_view,
                        "refine_mask": refine_mask,
                        "refined_view": refined_view,
                        "blended_view": blended_view,
                        "projected_update": projected_update,
                        "projected_update_mask": projected_update_mask,
                        "updated_panorama": refined,
                    },
                )

        if debug_writer:
            debug_writer(
                "final",
                {
                    "panorama": refined,
                },
            )

        return refined, records


class PromptTools:
    @staticmethod
    def normalize_prompt_mode(mode):
        mode = str(mode or "directional").strip().lower()

        if mode not in PROMPT_MODES:
            raise ValueError("invalid prompting.mode: " + mode)

        return mode

    @staticmethod
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
                array = ImageIO.to_uint8(array)
                pil_image = Image.fromarray(array).convert("RGB")
            pil_image.save(buffer, format="PNG")
            mime_type = "image/png"
            data = buffer.getvalue()

        encoded = base64.b64encode(data).decode("ascii")

        return "data:" + mime_type + ";base64," + encoded

    @staticmethod
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

        return json.loads(text[start: end + 1])

    @staticmethod
    def validate_schema(prompts):
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

    @staticmethod
    def save_prompts(path, prompts):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prompts = PromptTools.validate_schema(prompts)
        path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def save_prompt_payload(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = payload.copy()
        payload["mode"] = PromptTools.normalize_prompt_mode(payload.get("mode", "directional"))
        if "effective_prompts" not in payload:
            raise ValueError("prompt payload missing effective_prompts")
        payload["effective_prompts"] = PromptTools.validate_schema(payload["effective_prompts"])
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def effective_prompts_for_mode(mode, directional_prompts=None, caption_prompt=None):
        mode = PromptTools.normalize_prompt_mode(mode)

        if mode == "caption":
            caption_prompt = str(caption_prompt or "").strip()
            if not caption_prompt:
                raise ValueError("caption prompt is empty")

            return PromptTools.validate_schema(
                {
                    "scene_type": "uncertain",
                    "global_atmosphere_prompt": caption_prompt,
                    "sky_or_ceiling_prompt": caption_prompt,
                    "ground_or_floor_prompt": caption_prompt,
                    "negative_prompt": "",
                }
            )

        prompts = PromptTools.validate_schema(directional_prompts)
        if mode == "directional":
            return prompts

        prompts["sky_or_ceiling_prompt"] = prompts["ground_or_floor_prompt"] = prompts["global_atmosphere_prompt"]
        return prompts

    @staticmethod
    def prompts_to_anchored_synthesis_args(prompts):
        prompts = PromptTools.validate_schema(prompts)

        return (
            prompts["global_atmosphere_prompt"],
            prompts["sky_or_ceiling_prompt"],
            prompts["ground_or_floor_prompt"],
            prompts["negative_prompt"],
        )

    @staticmethod
    def completion_content(response):
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

    @staticmethod
    def create_completion(client, model, messages, extra_headers=None, json_mode=False):
        kwargs = dict(model=model, messages=messages, temperature=0.2)
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if json_mode:
            try:
                return client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
            except Exception:
                pass
        return client.chat.completions.create(**kwargs)

    @staticmethod
    def extra_headers(vlm_config):
        headers = {}
        http_referer = vlm_config.get("http_referer", "")
        title = vlm_config.get("title", "")

        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if title:
            headers["X-OpenRouter-Title"] = title

        return headers

    @staticmethod
    def _create_openai_client(vlm_config):
        from openai import OpenAI
        kwargs = {"api_key": vlm_config["api_key"]}
        if vlm_config.get("base_url"):
            kwargs["base_url"] = vlm_config["base_url"]
        return OpenAI(**kwargs)

    @staticmethod
    def generate_panorama_prompts(image, vlm_config):
        client = PromptTools._create_openai_client(vlm_config)
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": DEFAULT_USER_PROMPT},
                {"type": "image_url", "image_url": {"url": PromptTools.image_to_data_url(image)}},
            ]},
        ]
        response = PromptTools.create_completion(
            client,
            vlm_config["model"],
            messages,
            extra_headers=PromptTools.extra_headers(vlm_config),
            json_mode=True,
        )
        content = PromptTools.completion_content(response)
        prompts = PromptTools.extract_json_object(content)

        return PromptTools.validate_schema(prompts)

    @staticmethod
    def generate_caption_prompt(image, vlm_config):
        client = PromptTools._create_openai_client(vlm_config)
        messages = [
            {"role": "system", "content": DEFAULT_CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": DEFAULT_CAPTION_USER_PROMPT},
                {"type": "image_url", "image_url": {"url": PromptTools.image_to_data_url(image)}},
            ]},
        ]
        response = PromptTools.create_completion(
            client,
            vlm_config["model"],
            messages,
            extra_headers=PromptTools.extra_headers(vlm_config),
        )
        caption = re.sub(
            r"^```(?:text)?\s*|\s*```$",
            "",
            PromptTools.completion_content(response).strip(),
            flags=re.IGNORECASE,
        ).strip()
        if not caption:
            raise ValueError("caption prompt is empty")
        return caption


class DebugWriter:
    @staticmethod
    def format_yaw(yaw):
        yaw = int(round(yaw)) % 360

        return str(yaw).zfill(3)

    @staticmethod
    def format_pitch(pitch):
        pitch = int(round(pitch))
        sign = "p"
        if pitch < 0:
            sign = "n"

        return sign + str(abs(pitch)).zfill(3)

    @staticmethod
    def step_name(record):
        return (
                str(int(record["index"])).zfill(2)
                + "_"
                + record["phase"]
                + "_yaw_"
                + DebugWriter.format_yaw(record["yaw"])
                + "_pitch_"
                + DebugWriter.format_pitch(record["pitch"])
        )

    @staticmethod
    def panel_image(image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        return ImageIO.to_uint8(image)

    @staticmethod
    def stitch_panel_size(panels):
        square_sizes = []

        for _, image in panels:
            height, width = image.shape[:2]
            if width <= 0 or height <= 0:
                continue
            aspect = width / float(height)
            if aspect >= 0.75 and aspect <= 1.35:
                square_sizes.append(max(width, height))

        if square_sizes:
            size = max(square_sizes)
        else:
            size = 512

        size = max(320, min(size, 768))

        return size, size

    @staticmethod
    def add_label(image, label):
        cv2.putText(
            image,
            label.replace("_", " "),
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def save_stitch(path, payload, names):
        panels = []

        for name in names:
            if name in payload:
                panels.append((name, DebugWriter.panel_image(payload[name])))

        if not panels:
            return

        columns = 3
        label_height = 28
        padding = 8
        panel_width, panel_height = DebugWriter.stitch_panel_size(panels)
        rows = int(math.ceil(len(panels) / float(columns)))
        tile_height = panel_height + label_height
        canvas_width = columns * panel_width + (columns + 1) * padding
        canvas_height = rows * tile_height + (rows + 1) * padding
        canvas = np.full((canvas_height, canvas_width, 3), 24, dtype=np.uint8)

        for index, item in enumerate(panels):
            name, image = item
            row = index // columns
            column = index % columns
            left = padding + column * (panel_width + padding)
            top = padding + row * (tile_height + padding)
            tile = np.full((tile_height, panel_width, 3), 12, dtype=np.uint8)
            tile[:label_height, :, :] = 36
            DebugWriter.add_label(tile, name)

            image_height, image_width = image.shape[:2]
            scale = min(panel_width / float(image_width), panel_height / float(image_height))
            resized_width = max(1, int(round(image_width * scale)))
            resized_height = max(1, int(round(image_height * scale)))
            interpolation = cv2.INTER_LINEAR
            if scale < 1.0:
                interpolation = cv2.INTER_AREA
            resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
            image_left = (panel_width - resized_width) // 2
            image_top = label_height + (panel_height - resized_height) // 2
            tile[
                image_top: image_top + resized_height,
                image_left: image_left + resized_width,
            ] = resized
            canvas[top: top + tile_height, left: left + panel_width] = tile

        ImageIO.save_image(path, canvas)

    ANCHORED_SYNTHESIS_STITCH_PANELS = [
        "rendered_view",
        "view_known_mask",
        "raw_inpaint_mask",
        "inpaint_mask",
        "masked_view",
        "inpainted_view",
        "projected_update",
        "projected_update_mask",
        "projected_blend_mask",
        "updated_panorama",
        "updated_known_mask",
        "stitched_view_update",
        "stitched_view_update_mask",
        "stitched_view_blend_mask",
        "stitched_panorama",
        "stitched_known_mask",
    ]

    REFINEMENT_STITCH_PANELS = [
        "source_view",
        "refine_mask",
        "refined_view",
        "blended_view",
        "projected_update",
        "projected_update_mask",
        "updated_panorama",
    ]

    ANCHORED_SYNTHESIS_EVENTS = {
        "initial": [
            ("panorama", "00_initial_front_back_anchor"),
            ("known_mask", "01_initial_known_mask"),
            ("input_mask", "02_front_input_mask"),
            ("anchor_mask", "03_back_anchor_mask"),
        ],
        "after_front_back_vertical": [
            ("panorama", "30_after_front_back_vertical_panorama"),
            ("known_mask", "31_after_front_back_vertical_known_mask"),
        ],
        "after_horizontal": [
            ("panorama", "50_after_horizontal_panorama"),
            ("known_mask", "51_after_horizontal_known_mask"),
        ],
        "final": [
            ("panorama", "90_final_panorama"),
            ("known_mask", "91_final_known_mask"),
        ],
    }

    MISSING_MASK_PREFIXES = {
        "initial": "04",
        "after_front_back_vertical": "32",
        "after_horizontal": "52",
        "final": "92",
    }

    @staticmethod
    def save_anchored_synthesis_payload(output_dir, event, payload):
        if event == "step":
            DebugWriter.save_stitch(
                output_dir / "stitches" / (DebugWriter.step_name(payload["record"]) + ".png"),
                payload,
                DebugWriter.ANCHORED_SYNTHESIS_STITCH_PANELS,
            )
            return

        output_dir = Path(output_dir)
        for key, name in DebugWriter.ANCHORED_SYNTHESIS_EVENTS.get(event, []):
            if key in payload:
                ImageIO.save_image(output_dir / f"{name}.png", payload[key])

        if event in DebugWriter.MISSING_MASK_PREFIXES and "known_mask" in payload:
            prefix = DebugWriter.MISSING_MASK_PREFIXES[event]
            missing_mask = GeometryTools.compute_missing_mask(payload["known_mask"])
            ImageIO.save_image(output_dir / f"{prefix}_missing_mask.png", missing_mask)

        if event == "after_horizontal" and "stitched_panorama" in payload:
            ImageIO.save_image(output_dir / "53_after_horizontal_stitched_panorama.png", payload["stitched_panorama"])
            ImageIO.save_image(
                output_dir / "54_after_horizontal_stitched_known_mask.png",
                payload["stitched_known_mask"],
            )
        elif event == "final":
            if "context_panorama" in payload:
                ImageIO.save_image(output_dir / "93_context_panorama.png", payload["context_panorama"])
                ImageIO.save_image(output_dir / "94_context_known_mask.png", payload["context_known_mask"])

    @staticmethod
    def save_refinement_payload(output_dir, event, payload):
        output_dir = Path(output_dir)

        if event == "step":
            record = payload["record"]
            DebugWriter.save_stitch(
                output_dir / "refinement_stitches" / (DebugWriter.step_name(record) + ".png"),
                payload,
                DebugWriter.REFINEMENT_STITCH_PANELS,
            )
            return

        if event == "final":
            ImageIO.save_image(output_dir / "95_refined_panorama.png", payload["panorama"])
