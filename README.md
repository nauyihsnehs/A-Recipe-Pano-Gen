# A-Recipe-Pano-Gen
Project to implement A Recipe for Generating 3D Worlds From a Single Image, panorama generation part only

Use existing conda env `prmcam`

Use existing `stabilityai/stable-diffusion-2-inpainting`, `variant='fp16'`

Default local settings:
- input: `inputs/pano_aiypnfgomckrgf.jpg`
- panorama: `2048x1024`
- view size: `512`
- inpainting model: `stabilityai/stable-diffusion-2-inpainting`
- refinement model: `stabilityai/stable-diffusion-2-1-base`
- VLM endpoint: `http://127.0.0.1:11435/v1`
- VLM model: `qwen3.6`

project structure:
- [A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025.pdf](A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025.pdf): the original paper
- [reproduction-concept.md](reproduction-concept.md): the concept of reproduction, including the overall pipeline, the design of each module, and the implementation details
- [reproduction-plan.md](reproduction-plan.md): the detailed plan of reproduction with 5 stages
- [pipeline.py](pipeline.py): the end-to-end CLI entrypoint
- [pipeline.toml](pipeline.toml): default runtime configuration
- [pipeline_helper.py](pipeline_helper.py): consolidated utility and backend classes
- inputs/: example input images for testing
- outputs/: example output folder

## Run pipeline

Use PowerShell with the existing environment:

```powershell
& 'D:\_conda_envs\prmcam\python.exe' pipeline.py
```

The default configuration is loaded from `pipeline.toml`. Use another config file with `--config`:

```powershell
& 'D:\_conda_envs\prmcam\python.exe' pipeline.py --config pipeline.toml
```

Each run writes to a timestamped folder under `outputs/`.
Final output is written to `outputs/YYYYmmdd_HHMMSS/<input_name>_pano.png`.
Debug artifacts are written to `outputs/YYYYmmdd_HHMMSS/debug/`.

## Notes

- Prompt generation runs on every pipeline run and writes `debug/prompts.json`.
- The VLM client uses the OpenAI SDK and can target OpenAI-compatible endpoints, including OpenRouter-style providers configured in `[vlm]` with optional `http_referer` and `title`.
- Diffusers runs on CUDA with float16 and allows model downloads. Model loading first tries `variant='fp16'`, then retries without `variant`.
- Outputs, inputs, image files, and model files are intentionally ignored by git.
- The implementation covers panorama generation only, not later 3D reconstruction stages.
