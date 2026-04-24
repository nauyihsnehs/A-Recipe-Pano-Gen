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
- [pipeline_helper.py](pipeline_helper.py): consolidated pipeline implementation
- inputs/: example input images for testing
- outputs/: example output folder

## Run pipeline

Use PowerShell with the existing environment:

```powershell
& 'D:\_conda_envs\prmcam\python.exe' pipeline.py
```

The default configuration is loaded from `pipeline.toml`:

```powershell
& 'D:\_conda_envs\prmcam\python.exe' pipeline.py --config pipeline.toml
```

Useful options:

```powershell
& 'D:\_conda_envs\prmcam\python.exe' pipeline.py --generate-prompts
& 'D:\_conda_envs\prmcam\python.exe' pipeline.py --disable-refinement
& 'D:\_conda_envs\prmcam\python.exe' pipeline.py --output outputs/final/final_pano.png
```

Final output is written to `outputs/final/final_pano.png`.
Debug artifacts are written to `outputs/final/debug/`.

## Notes

- Prompt generation requires the local OpenAI-compatible VLM server to be running on `127.0.0.1:11435`.
- Outputs, inputs, image files, and model files are intentionally ignored by git.
- The implementation covers panorama generation only, not later 3D reconstruction stages.
