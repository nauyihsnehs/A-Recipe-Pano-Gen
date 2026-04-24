import numpy as np
from PIL import Image


def _to_uint8_image(image):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _to_mask_image(mask):
    return np.where(mask > 0, 255, 0).astype(np.uint8)


class DiffusersInpaintingBackend:
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
        from diffusers import StableDiffusionInpaintPipeline, DiffusionPipeline

        device = self._resolve_device()
        torch_dtype = self._resolve_torch_dtype(device)
        # self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, variant='fp16',
            local_files_only=self.local_files_only, safety_checker=None
        )
        self.pipeline = self.pipeline.to(device)

        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()

        self.device = device

        return self.pipeline

    def __call__(
        self,
        image,
        mask,
        prompt,
        negative_prompt=None,
        seed=42,
        num_steps=40,
        guidance_scale=7.5,
    ):
        import torch

        if self.pipeline is None:
            self.load()

        image = _to_uint8_image(image)
        mask = _to_mask_image(mask)
        image_pil = Image.fromarray(image).convert("RGB")
        mask_pil = Image.fromarray(mask).convert("L")
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


def run_inpainting(image, mask, prompt, negative_prompt, seed, backend, num_steps, guidance_scale):
    return backend(
        image,
        mask,
        prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
    )
