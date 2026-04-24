import cv2
import numpy as np
from PIL import Image

from projection import (
    project_perspective_mask_to_equirect,
    project_perspective_to_equirect,
    render_perspective_from_equirect,
    render_perspective_mask_from_equirect,
)
from view_scheduler import anchored_view_schedule


def _to_uint8_image(image):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _soft_alpha(mask, mask_blur):
    alpha = np.where(mask > 0, 1.0, 0.0).astype(np.float32)

    if mask_blur and mask_blur > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), float(mask_blur))
        alpha = np.clip(alpha, 0.0, 1.0)

    return alpha[..., None]


class DiffusersImg2ImgRefinementBackend:
    def __init__(self, model_id, device=None, torch_dtype=None, local_files_only=False):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only
        self.pipeline = None

    def _resolve_device(self):
        import torch

        if self.device and self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def _resolve_torch_dtype(self, device):
        import torch

        if self.torch_dtype is None or self.torch_dtype == "auto":
            if str(device).startswith("cuda"):
                return torch.float16

            return torch.float32

        if self.torch_dtype == "float16":
            return torch.float16
        if self.torch_dtype == "float32":
            return torch.float32
        if self.torch_dtype == "bfloat16":
            return torch.bfloat16

        return self.torch_dtype

    def load(self):
        from diffusers import StableDiffusionImg2ImgPipeline

        device = self._resolve_device()
        torch_dtype = self._resolve_torch_dtype(device)
        kwargs = {
            "torch_dtype": torch_dtype,
            "local_files_only": self.local_files_only,
            "safety_checker": None,
        }

        if str(device).startswith("cuda"):
            kwargs["variant"] = "fp16"

        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            **kwargs,
        )
        self.pipeline = self.pipeline.to(device)

        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()

        self.device = device

        return self.pipeline

    def __call__(
        self,
        image,
        prompt,
        negative_prompt=None,
        seed=42,
        num_steps=30,
        guidance_scale=7.5,
        denoise_strength=0.3,
    ):
        import torch

        if self.pipeline is None:
            self.load()

        image = _to_uint8_image(image)
        image_pil = Image.fromarray(image).convert("RGB")
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=image_pil,
            strength=float(denoise_strength),
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )

        return np.asarray(result.images[0].convert("RGB"))


def _project_refined_view(panorama, refined_view, refine_mask, view, input_mask):
    pano_size = (panorama.shape[1], panorama.shape[0])
    projected_view, footprint = project_perspective_to_equirect(
        refined_view,
        view["fov_x"],
        view["fov_y"],
        view["yaw"],
        view["pitch"],
        pano_size,
    )
    projected_mask = project_perspective_mask_to_equirect(
        refine_mask,
        view["fov_x"],
        view["fov_y"],
        view["yaw"],
        view["pitch"],
        pano_size,
    )
    projected_update_mask = np.where(
        (projected_mask > 0) & (footprint > 0) & (input_mask == 0),
        255,
        0,
    ).astype(np.uint8)

    output = panorama.copy()
    projected_update = np.zeros_like(panorama)
    region = projected_update_mask > 0
    output[region] = projected_view[region]
    projected_update[region] = projected_view[region]

    return output, projected_update, projected_update_mask


def refine_panorama(
    panorama,
    generated_mask,
    input_mask,
    prompt,
    backend,
    negative_prompt="",
    seed=42,
    num_steps=30,
    guidance_scale=7.5,
    denoise_strength=0.3,
    mask_blur=32,
    view_size=512,
    middle_fov=85.0,
    vertical_fov=120.0,
    debug_writer=None,
):
    refined = panorama.copy()
    records = []
    schedule = anchored_view_schedule(middle_fov=middle_fov, vertical_fov=vertical_fov)

    for index, view in enumerate(schedule):
        out_size = (view_size, view_size)
        view_generated_mask = render_perspective_mask_from_equirect(
            generated_mask,
            view["yaw"],
            view["pitch"],
            view["fov_x"],
            view["fov_y"],
            out_size,
        )
        view_input_mask = render_perspective_mask_from_equirect(
            input_mask,
            view["yaw"],
            view["pitch"],
            view["fov_x"],
            view["fov_y"],
            out_size,
        )
        refine_mask = np.where(
            (view_generated_mask > 0) & (view_input_mask == 0),
            255,
            0,
        ).astype(np.uint8)
        pixel_count = int(np.count_nonzero(refine_mask))
        record = {
            "index": index,
            "phase": view["phase"],
            "yaw": view["yaw"],
            "pitch": view["pitch"],
            "refine_pixels": pixel_count,
        }

        if pixel_count == 0:
            record["skipped"] = True
            records.append(record)
            continue

        source_view = render_perspective_from_equirect(
            refined,
            view["yaw"],
            view["pitch"],
            view["fov_x"],
            view["fov_y"],
            out_size,
        )
        refined_view = backend(
            source_view,
            prompt,
            negative_prompt=negative_prompt,
            seed=int(seed) + index,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            denoise_strength=denoise_strength,
        )
        alpha = _soft_alpha(refine_mask, mask_blur)
        blended_view = (
            refined_view.astype(np.float32) * alpha
            + source_view.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
        refined, projected_update, projected_update_mask = _project_refined_view(
            refined,
            blended_view,
            refine_mask,
            view,
            input_mask,
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
