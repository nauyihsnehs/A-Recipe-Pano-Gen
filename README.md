# A-Recipe-Pano-Gen
Project to implement A Recipe for Generating 3D Worlds From a Single Image, panorama generation part only

Use existing conda env `prmcam`

Use existing `stabilityai/stable-diffusion-2-inpainting`, `variant='fp16'`

Default local settings:
- input: `inputs/pano_aiypnfgomckrgf.jpg`
- panorama: `2048x1024`
- view size: `512`
- inpainting model: `stabilityai/stable-diffusion-2-inpainting`
- refinement model: `stabilityai/stable-diffusion-2-base`
- VLM endpoint: `http://127.0.0.1:11435/v1`
- VLM model: `qwen3.6`

project structure:
- [A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025.pdf](A_Recipe_for_Generating_3D_Worlds_from_a_Single_Image_ICCV_2025.pdf): the original paper
- [reproduction-concept.md](reproduction-concept.md): the concept of reproduction, including the overall pipeline, the design of each module, and the implementation details
- [reproduction-plan.md](reproduction-plan.md): the detailed plan of reproduction with 5 stages
- inputs/: example input images for testing
- outputs/: example output folder

## Run stages

Use PowerShell with the existing environment:

```powershell
& 'D:\_conda_envs\prmcam\python.exe' run_stage1_geometry.py
& 'D:\_conda_envs\prmcam\python.exe' run_stage2_roundtrip.py --local-files-only
& 'D:\_conda_envs\prmcam\python.exe' run_stage4_prompts.py
& 'D:\_conda_envs\prmcam\python.exe' run_stage3_anchored.py
```

Stage 3 reads `outputs/stage4/prompts.json` by default when it exists.

## End-to-end run

```powershell
& 'D:\_conda_envs\prmcam\python.exe' run_recipe_pano.py
```

Useful options:

```powershell
& 'D:\_conda_envs\prmcam\python.exe' run_recipe_pano.py --generate-prompts
& 'D:\_conda_envs\prmcam\python.exe' run_recipe_pano.py --disable-refinement
& 'D:\_conda_envs\prmcam\python.exe' run_recipe_pano.py --output outputs/final/final_pano.png
```

Final output is written to `outputs/final/final_pano.png`.
Debug artifacts are written to `outputs/final/debug/`.

## Notes

- Stage 4 requires the local OpenAI-compatible VLM server to be running on `127.0.0.1:11435`.
- Outputs, inputs, image files, and model files are intentionally ignored by git.
- The implementation covers panorama generation only, not later 3D reconstruction stages.
