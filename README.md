# A-Recipe-Pano-Gen

Unofficial implementation of the panorama-generation part of:

**A Recipe for Generating 3D Worlds From a Single Image**

- Paper: [arXiv:2503.16611](https://arxiv.org/abs/2503.16611)
- Project page: [katjaschwarz.github.io/worlds](https://katjaschwarz.github.io/worlds/)
- Local PDF: [A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025.pdf](A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025.pdf)
- README structure reference: [PanoFree-Unofficial](https://github.com/nauyihsnehs/PanoFree-Unofficial/blob/main/README.md)

This repo currently implements a partial reproduction of Section 3.1, panorama generation. It does not implement the paper's point-cloud-conditioned inpainting, 3D Gaussian Splatting reconstruction, or VR scene export.

## Status

Implemented:

- Perspective image projection into a 2:1 equirectangular panorama.
- Perspective rendering from an equirectangular panorama.
- Anchored panorama synthesis:
  - front input projection,
  - temporary backside anchor,
  - top and bottom generation first,
  - anchor removal,
  - 8-view horizontal sweep.
- VLM-generated directional prompts:
  - global atmosphere,
  - sky or ceiling,
  - ground or floor,
  - negative prompt.
- Configurable prompt modes:
  - `directional` default,
  - `coarse`,
  - `caption`.
- Soft inpainting mask edge blur for main synthesis.
- Conservative overlap blending for generated regions.
- Shared stage-level Diffusers generator seeded once per synthesis/refinement stage.
- Public Diffusers inpainting backend.
- Optional partial-denoising refinement backend.
- Debug artifacts for masks, views, projected updates, prompts, and final outputs.

Known differences from the paper:

- Dust3R FoV estimation is not implemented. `input_fov_x` is configured manually.
- The proprietary T2I inpainting model with ControlNet conditioning is replaced with a public Diffusers model.
- Default view size is `512`, while the paper uses `1024` px square inpainting views.
- Default panorama size is `2048x1024`, while paper-scale output is `4096x2048`.
- Ad-hoc and sequential panorama baselines are not implemented.
- Florence-2 is not implemented; the optional `caption` prompt mode uses the configured VLM endpoint.
- The 3D world stages are not implemented.
- Quantitative experiments and paper baselines are not implemented.

See [reproduction-report.md](reproduction-report.md) for the full alignment review.

## File And Directory Overview

```text
.
├── inputs/                         # Local input images, ignored by git
├── outputs/                        # Timestamped runs and debug artifacts, ignored by git
├── A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025.pdf
├── pipeline.py                     # CLI orchestration
├── pipeline_helper.py              # Geometry, scheduling, model backends, prompts, debug writing
├── pipeline.toml                   # Default local runtime config
├── reproduction-concept.md          # Earlier method notes
├── reproduction-plan.md             # Earlier staged reproduction plan
├── reproduction-report.md           # Current paper alignment report
└── README.md
```

## Requirements

Use the existing conda environment:

```text
D:\_conda_envs\prmcam
```

Main Python dependencies used by the current code:

```text
numpy
Pillow
opencv-python
torch
diffusers
openai
```

CUDA is required by the Diffusers backends. Model loading first tries `variant="fp16"` and falls back to loading without the variant.

## Inference

Run with the default config:

```powershell
& 'D:\_conda_envs\prmcam\python.exe' pipeline.py
```

Run with an explicit config path:

```powershell
& 'D:\_conda_envs\prmcam\python.exe' pipeline.py --config pipeline.toml
```

Each run writes a timestamped directory under `outputs/`.

Final panorama:

```text
outputs/YYYYmmdd_HHMMSS/<input_name>_pano.png
```

Debug directory:

```text
outputs/YYYYmmdd_HHMMSS/debug/
```

## Config

The pipeline reads TOML from `pipeline.toml`.

Top-level sections:

- `paths`: input image and output root.
- `panorama`: equirectangular output size and input horizontal FoV.
- `view`: rendered perspective view size and stage FoVs.
- `models`: Diffusers model ids for inpainting and refinement.
- `vlm`: OpenAI-compatible VLM endpoint and model settings.
- `prompting`: prompt mode, one of `directional`, `coarse`, or `caption`.
- `synthesis`: seed, inpainting step count, guidance scale, mask dilation, mask blur, and overlap blending.
- `refinement`: optional img2img refinement settings.

Current default local settings:

```text
input: inputs/symmetrical_garden_02_8k.jpg
panorama: 2048x1024
view size: 512
input fov_x: 90.0
middle fov: 85.0
vertical fov: 120.0
inpainting model: stabilityai/stable-diffusion-2-inpainting
refinement model: stabilityai/stable-diffusion-2-1-base
VLM endpoint: http://127.0.0.1:11435/v1
VLM model: qwen3.6
prompt mode: directional
synthesis mask blur: 8.0
synthesis mask dilation: 5 px kernel, 1 iteration
overlap blending: true
```

## Debug Outputs

Prompt generation runs on every pipeline run and writes:

```text
debug/prompts.json
```

Anchored synthesis writes files under:

```text
debug/anchored/
```

Main debug artifacts include:

- initial front/back anchor panorama,
- known masks and missing masks,
- per-view rendered crops,
- per-view raw inpainting masks,
- per-view inpainting masks,
- per-view soft inpainting masks,
- masked views,
- inpainted views,
- projected panorama updates,
- projected overlap blend masks,
- final stage-3 panorama,
- generated mask,
- input preserve mask,
- final panorama.

When refinement is enabled, per-view refinement artifacts are written under:

```text
debug/refinement_steps/
```

## Citation

```bibtex
@article{schwarz2025recipe,
  title={A Recipe for Generating 3D Worlds From a Single Image},
  author={Schwarz, Katja and Rozumnyi, Denys and Rota Bulo, Samuel and Porzi, Lorenzo and Kontschieder, Peter},
  journal={arXiv preprint arXiv:2503.16611},
  year={2025}
}
```
