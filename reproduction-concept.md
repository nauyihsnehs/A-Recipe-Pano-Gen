# Reproduction Plan: `A Recipe` — Section 3.1 Panorama Generation

## Scope

This reproduction targets only **Section 3.1: Panorama Generation** from *A Recipe for Generating 3D Worlds From a Single Image*.

The goal is to reproduce the **methodological behavior** of the panorama synthesis stage, not the full experimental setting, quantitative metrics, 3D lifting, point-cloud-conditioned inpainting, Gaussian Splatting reconstruction, or VR rendering.

The reproduction should therefore implement a pipeline that:

1. Takes a single perspective input image.
2. Estimates or accepts its field of view.
3. Embeds the input image into an equirectangular panorama canvas.
4. Progressively outpaints missing regions by rendering overlapping perspective views.
5. Uses the paper's **anchored panorama synthesis heuristic** as the target method.
6. Uses a VLM to produce panorama-level, sky/ceiling, and ground/floor prompts.
7. Optionally applies a final partial-denoising refinement pass.

---

## 1. Method Decomposition

### 1.1 Input Assumptions

Input:

- A single RGB perspective image `I`.
- The image is assumed to depict the initial forward-facing view.
- If camera intrinsics or FoV are unavailable, estimate horizontal FoV using a pretrained camera / geometry estimator, following the paper's use of Dust3R.
- Vertical FoV is derived under the equal-focal-length assumption.

Expected output:

- A full equirectangular panorama image, typically with aspect ratio `2:1`.
- The panorama should preserve the appearance of the input view and synthesize plausible sky/ceiling, ground/floor, and side/back regions.

Out of scope:

- 3D point cloud lifting.
- Metric depth estimation.
- Point-cloud-conditioned ControlNet fine-tuning.
- Gaussian Splatting reconstruction.
- Quantitative evaluation.

---

### 1.2 Core Formulae to Implement

The input image is projected into an equirectangular panorama by mapping normalized perspective image coordinates `(u, v) ∈ [-1, 1]` to spherical angles:

```text
theta = u * fov_x / 2
phi   = v * fov_y / 2
````

Then spherical angles are mapped to equirectangular coordinates:

```text
x = ((theta + pi) / (2 * pi)) * W
y = ((phi + pi / 2) / pi) * H
```

where:

* `W, H` are the panorama width and height.
* `W:H = 2:1`.
* `fov_x` is estimated or user-provided.
* `fov_y` is derived assuming equal focal length along x and y axes.

Implementation should provide reusable functions:

```text
perspective_to_equirectangular()
equirectangular_to_perspective()
render_perspective_view()
paste_or_project_view_to_panorama()
compute_known_mask()
compute_inpaint_mask()
```

---

## 2. Target Reproduction Strategy

### 2.1 Heuristic to Reproduce

The paper investigates three heuristics:

1. **Ad-hoc**

    * Directly asks the model to synthesize an equirectangular panorama.
    * This is not the target method because it does not reliably produce correct sky/ground distortion.

2. **Sequential**

    * Rotates right and left first, then fills sky and ground.
    * This can produce a coherent middle band, but sky/ground may not match the scene due to lack of global context.

3. **Anchored**

    * Duplicates the input image to the backside.
    * Generates sky and ground first.
    * Removes the backside anchor.
    * Rotates around the horizontal band to fill the remaining panorama.
    * This is the target method because the paper reports that anchoring provides global context and improves coherent equirectangular synthesis.

The reproduction should implement **Anchored** as the main pipeline.
`Ad-hoc` and `Sequential` may be implemented only as debugging baselines.

---

## 3. Modular Implementation Plan

## Stage 1 — Projection and Panorama Canvas

### Goal

Build the geometric infrastructure required by the method.

### Required Modules

```text
camera.py
projection.py
canvas.py
mask.py
```

### Functions

```text
estimate_or_load_fov(input_image) -> fov_x, fov_y
create_equirectangular_canvas(width, height) -> panorama, known_mask
project_input_to_equirectangular(input_image, fov_x, fov_y, yaw=0, pitch=0)
render_perspective_from_equirectangular(panorama, yaw, pitch, fov, resolution)
update_panorama_from_generated_view(view, yaw, pitch, fov)
```

### Default Settings

```yaml
panorama_resolution: 4096x2048
inpainting_view_resolution: 1024x1024
input_yaw: 0
input_pitch: 0
input_roll: 0
```

### Verification Milestone

At the end of this stage:

* The input image can be projected into the center of an equirectangular panorama.
* Perspective views rendered from the panorama should approximately reconstruct the input view.
* The known/unknown mask should be visually correct.
* No diffusion model is required yet.

### Difficulty

Medium.

### Mature References

* `py360convert`
* custom OpenCV / NumPy spherical projection
* common panorama projection utilities

### Missing or Ambiguous Details

* The paper does not specify exact implementation details for interpolation, mask rasterization, or blending.
* Use bilinear sampling for images and nearest-neighbor sampling for masks.
* Use hard generated-region masks for compositing.

---

## Stage 2 — Basic Inpainting View Loop

### Goal

Implement a generic progressive outpainting loop over rendered perspective crops.

### Required Modules

```text
view_scheduler.py
inpaint_runner.py
panorama_update.py
```

### Core Procedure

For each target camera view:

1. Render a perspective crop from the current panorama.
2. Compute the missing region mask.
3. Send the masked crop, mask, and prompt to the inpainting model.
4. Receive the completed crop.
5. Project the completed crop back into the panorama.
6. Update the panorama and known mask.

### Default Inpainting Model

The paper uses a proprietary transformer-based T2I inpainting model with ControlNet-like masked-image conditioning. For reproduction, use a publicly available substitute:

```yaml
preferred_public_substitutes:
  - Stable Diffusion XL Inpainting
  - SD 2.0 / SD 2.1 Inpainting
  - FLUX inpainting, if available
  - any diffusion inpainting model that accepts image + mask + text
```

### Verification Milestone

At the end of this stage:

* A single missing perspective crop can be completed.
* The generated crop can be projected back into the panorama.
* The known mask grows after each step.
* The system can run with a dummy prompt before implementing VLM prompts.

### Difficulty

Medium to high.

### Mature References

* Diffusers inpainting pipelines.
* Stable Diffusion inpainting examples.
* Perspective-to-panorama projection code from panorama generation projects.

### Missing or Ambiguous Details

* The paper does not specify the exact proprietary model architecture.
* Public models may differ substantially in prompt adherence and mask behavior.
* Exact sampler, guidance scale, number of steps, and denoising strength are not specified in Sec. 3.1.

Recommended reproducible defaults:

```yaml
num_inference_steps: 30-50
guidance_scale: 5.0-8.0
mask_dilate_kernel: 5
seed_control: enabled
output_resolution: 1024x1024
```

---

## Stage 3 — Anchored Panorama Synthesis

### Goal

Implement the paper's main panorama synthesis heuristic.

### Target Algorithm

```text
1. Project the input image to the front of the equirectangular panorama.
2. Duplicate the input image to the backside as a temporary anchor.
3. Generate the top / sky / ceiling region.
4. Generate the bottom / ground / floor region.
5. Remove the backside anchor from the known mask.
6. Generate the horizontal panorama band by rotating around the camera.
7. Produce the final 360-degree equirectangular panorama.
```

### View Schedule

Use the settings reported in the paper:

```yaml
middle_band:
  num_views: 8
  fov: 85 degrees
  pitch: 0 degrees
  yaw_values: evenly spaced around 360 degrees

top_region:
  num_views: 4
  fov: 120 degrees
  pitch: +90 degrees or upward-looking equivalent
  yaw_values: [ 0, 90, 180, 270 ]

bottom_region:
  num_views: 4
  fov: 120 degrees
  pitch: -90 degrees or downward-looking equivalent
  yaw_values: [ 0, 90, 180, 270 ]
```

### Important Implementation Detail

The backside duplicate is not final content. It is a temporary visual-context anchor. After generating sky and ground, remove the backside anchor from the known mask before generating the horizontal band.

### Verification Milestone

At the end of this stage:

* The system produces a complete equirectangular panorama.
* Sky/ceiling and ground/floor should be more globally consistent than the sequential baseline.
* The backside should not contain a visible duplicate of the input image after the horizontal pass.
* Seam artifacts may still exist but should not completely break the panorama.

### Difficulty

High.

### Mature References

* Spherical view rendering.
* Diffusion inpainting.
* Existing panorama stitching utilities.

### Missing or Ambiguous Details

The paper does not fully specify:

* The exact yaw ordering for the 8 horizontal views.
* Whether the generated views are blended immediately or after all views are generated.
* How overlapping generated regions are weighted.
* How the temporary backside anchor is masked out.
* Whether top/bottom generation uses strict pitch `±90°` or multiple tilted views near the poles.

Recommended implementation:

```yaml
horizontal_yaw_order:
  - 0
  - 45
  - 90
  - 135
  - 180
  - 225
  - 270
  - 315

top_bottom_yaw_order:
  - 0
  - 90
  - 180
  - 270

blend_policy:
  known_input_pixels: highest priority
  newly_inpainted_pixels: weighted by view-center confidence
  overlap: feathered alpha blending
```

---

## Stage 4 — VLM Prompt Generation

### Goal

Reproduce the paper's prompt-generation logic.

### Paper-Level Behavior

The paper reports that a simple caption of the input image is insufficient because it causes duplication of the input content. A coarse prompt helps reduce duplication but may still cause sky and ground to duplicate the scene. Therefore, the method uses a VLM to generate three prompts:

1. A coarse scene-atmosphere prompt that ignores central objects and people.
2. A sky or ceiling prompt.
3. A ground or floor prompt.

The VLM also infers whether the scene is indoor or outdoor.

### Required Module

```text
prompt_generator.py
```

### Prompt Schema

```json
{
  "scene_type": "indoor | outdoor | uncertain",
  "global_atmosphere_prompt": "...",
  "sky_or_ceiling_prompt": "...",
  "ground_or_floor_prompt": "...",
  "negative_prompt": "... optional ..."
}
```

### Recommended VLM Instruction

```text
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
```

### Prompt Usage

```yaml
top_views:
  positive_prompt: sky_or_ceiling_prompt + global_atmosphere_prompt

bottom_views:
  positive_prompt: ground_or_floor_prompt + global_atmosphere_prompt

horizontal_views:
  positive_prompt: global_atmosphere_prompt

negative_prompt:
  shared across all views
```

### Verification Milestone

At the end of this stage:

* The same input image should generate different prompts for global, top, and bottom regions.
* Foreground objects in the input image should not be repeatedly synthesized across all views.
* Indoor scenes should use ceiling/floor prompts.
* Outdoor scenes should use sky/ground prompts.

### Difficulty

Low to medium.

### Mature References

* Llama 3.2 Vision, GPT-4o, Qwen-VL, InternVL, Florence-2 for captioning.
* L-MAGIC-style view-level prompt decomposition as a related implementation reference.

### Missing or Ambiguous Details

The paper does not provide the exact VLM prompt template in the main text.
Therefore, the reproduction should treat prompt wording as an implementation detail while preserving the paper-level behavior.

---

## Stage 5 — Final Refinement and Packaging

### Goal

Apply the paper's optional final refinement pass and package the pipeline into a reproducible CLI.

### Refinement Behavior

The paper applies a partial denoising process to improve image quality:

```text
Use a standard text-to-image diffusion model.
Denoise using the last 30% of the time steps.
Blend the refined result into the panorama using the generated-region mask.
```

### Implementation

```text
refine_panorama(
    panorama,
    generated_mask,
    prompt=global_atmosphere_prompt,
    denoise_strength=0.3,
    protect_mask=None
)
```

### CLI Target

```bash
python run_recipe_pano.py \
  --input input.jpg \
  --output output_pano.png \
  --pano-width 4096 \
  --pano-height 2048 \
  --view-size 1024 \
  --fov-mode estimate \
  --method anchored \
  --seed 42
```

### Verification Milestone

At the end of this stage:

* The pipeline can run end-to-end from one input image to one equirectangular panorama.
* Intermediate debug outputs are saved:

    * projected input panorama
    * known masks
    * rendered masked views
    * inpainted views
    * panorama after top/bottom generation
    * panorama after horizontal generation
    * final refined panorama
* The final result should be visually inspectable in a panorama viewer.

### Difficulty

Medium.

### Missing or Ambiguous Details

* The paper does not specify the exact refinement model.
* It does not define the exact mask compositing kernel.
* It does not state whether refinement is applied to the full panorama at once or in rendered perspective crops.

Recommended implementation:

```yaml
default_refinement:
  mode: perspective_crop_refinement
  denoise_strength: 0.3
```

---

## 4. Reproduction Feasibility

| Module                                   | Feasibility | Existing / Mature References                      | External Dependencies                      | Main Risk                                                       |
|------------------------------------------|------------:|---------------------------------------------------|--------------------------------------------|-----------------------------------------------------------------|
| Perspective ↔ equirectangular projection |        High | py360convert, OpenCV, NumPy, custom grid sampling | None beyond image libraries                | Coordinate convention mistakes                                  |
| FoV estimation                           |      Medium | Dust3R / DUSt3R-like camera estimation            | Heavy model dependency                     | Estimated FoV may be unstable                                   |
| Masked perspective rendering             |        High | Standard spherical projection                     | None                                       | Mask boundary aliasing                                          |
| Diffusion inpainting                     |        High | Diffusers inpainting pipelines                    | GPU, pretrained model weights              | Public model differs from paper's proprietary model             |
| Anchored synthesis heuristic             |      Medium | Implementable from paper description              | Projection + inpainting                    | Backside anchor removal and overlap blending are underspecified |
| VLM prompt generation                    |        High | GPT-4o, Llama Vision, Qwen-VL, InternVL           | VLM API or local VLM                       | Prompt wording not provided exactly                             |
| Final partial denoising refinement       |      Medium | img2img / inpainting diffusion                    | Diffusion model                            | Exact refinement protocol is underspecified                     |
| Full paper-quality reproduction          |  Medium-low | Requires model tuning and prompt tuning           | Strong GPU + high-quality inpainting model | Main paper uses proprietary model                               |

---

## 5. Known Underspecified Parts

The following details are insufficiently specified in the paper and should be treated as implementation choices:

1. Exact inpainting model architecture and weights.
2. Exact ControlNet conditioning format.
3. Exact sampler, guidance scale, step count, and random seed policy.
4. Exact VLM prompt template.
5. Exact FoV estimation implementation and fallback policy.
6. Exact yaw/pitch schedule for all rendered views.
7. Exact overlap blending strategy.
8. Exact procedure for removing the backside anchor.
9. Exact mask dilation and compositing settings.
10. Exact final refinement implementation.
11. Whether refinement is performed on the panorama directly or via perspective crops.

For faithful reproduction, these should be explicitly recorded in the implementation config.

---

## 6. Suggested Implementation Defaults

```yaml
panorama:
  width: 4096
  height: 2048
  aspect_ratio: 2:1

input:
  assumed_yaw: 0
  assumed_pitch: 0
  assumed_roll: 0
  fov_estimator: dust3r_or_manual
  fallback_fov_x: 70

inpainting:
  view_resolution: 1024
  model: public_t2i_inpainting_model
  num_steps: 40
  guidance_scale: 7.0
  mask_dilate_kernel: 5
  protect_conditioning_anchor: true

anchored_synthesis:
  use_backside_anchor: true
  generate_vertical_regions_first: true
  remove_backside_anchor_before_horizontal_generation: true

middle_band:
  num_views: 8
  fov: 85
  pitch: 0
  yaw_values: [ 0, 45, 90, 135, 180, 225, 270, 315 ]

top_region:
  num_views: 4
  fov: 120
  pitch: 90
  yaw_values: [ 0, 90, 180, 270 ]

bottom_region:
  num_views: 4
  fov: 120
  pitch: -90
  yaw_values: [ 0, 90, 180, 270 ]

prompting:
  vlm_required: true
  prompts:
    - global_atmosphere_prompt
    - sky_or_ceiling_prompt
    - ground_or_floor_prompt
    - negative_prompt

refinement:
  enabled: true
  denoise_strength: 0.3
```

---

## 7. LLM Code Agent Cost-Saving Strategy

### 7.1 Avoid One-Shot Full Pipeline Coding

Do not ask the coding agent to implement the entire method at once.
Instead, request implementation by milestone:

1. Projection and mask visualization.
2. Single-view inpainting round trip.
3. Anchored top/bottom generation.
4. Horizontal sweep generation.
5. VLM prompting and final refinement.

Each milestone should have a concrete visual output and a small debug script.

---

### 7.2 Use Fixed Interfaces

Before asking the agent to code, define stable function signatures:

```python
project_perspective_to_equirect(
    image, fov_x, fov_y, yaw, pitch, pano_size
)

render_perspective_from_equirect(
    pano, yaw, pitch, fov, out_size
)

run_inpainting(
    image, mask, prompt, negative_prompt, seed
)

update_pano_with_view(
    pano, known_mask, generated_view, view_mask, yaw, pitch, fov
)
```

This prevents repeated redesign of the codebase.

---

### 7.3 Require Debug Artifacts at Every Stage

The agent should save:

```text
debug/
  00_projected_input.png
  01_known_mask.png
  02_rendered_view_yaw_*.png
  03_inpaint_mask_yaw_*.png
  04_inpainted_view_yaw_*.png
  05_updated_pano_step_*.png
  06_final_pano.png
```

This reduces unnecessary follow-up calls because visual failures become easy to locate.

---

### 7.4 Keep the Diffusion Backend Swappable

Implement a minimal backend interface:

```python
class InpaintingBackend:
    def __call__(self, image, mask, prompt, negative_prompt, seed):
        ...
```

Then the pipeline can switch between:

```text
dummy color fill
OpenCV inpainting
Stable Diffusion inpainting
SDXL inpainting
FLUX inpainting
proprietary model API
```

This allows geometry debugging without spending GPU calls.

---

### 7.5 Separate Geometry From Generation

Token- and call-efficient coding order:

```text
First: geometry-only pipeline with dummy fill.
Second: real inpainting backend.
Third: VLM prompt generation.
Fourth: refinement.
```

Do not debug projection and diffusion behavior simultaneously.

---

### 7.6 Use Config Files Instead of Hard-Coded Parameters

Create one `recipe_pano.yaml`:

```yaml
pano_width: 4096
pano_height: 2048
view_size: 1024
middle_fov: 85
vertical_fov: 120
num_middle_views: 8
num_vertical_views: 4
seed: 42
```

Then future changes require no code rewrite.

---

## 8. Minimal Acceptance Criteria

A reproduction should be considered method-complete when it satisfies the following:

1. The input image is correctly embedded into an equirectangular panorama.
2. The anchored heuristic is implemented:

    * backside anchor inserted,
    * top/bottom generated first,
    * backside anchor removed,
    * horizontal band generated afterwards.
3. The system uses separate global, sky/ceiling, and ground/floor prompts.
4. The default schedule uses:

    * 8 middle views,
    * 85° FoV for the middle band,
    * 4 top views and 4 bottom views,
    * 120° FoV for vertical regions.
5. The pipeline saves intermediate views and masks.
6. The final output is a full 2:1 equirectangular panorama.
7. The implementation does not include or depend on Sec. 3.2 or Sec. 3.3.

---

## 9. Recommended Milestone Checklist

### Milestone 1: Geometry Sanity Check

Expected result:

* Input image appears at the correct front-facing position.
* Rendering the front view from the panorama approximately reconstructs the input.
* Masks are correct.

Pass condition:

```text
projection_error_is_visually_small = true
known_mask_matches_input_region = true
```

---

### Milestone 2: Single Inpainting Round Trip

Expected result:

* One masked perspective crop is generated.
* The crop can be projected back into the panorama.
* No major coordinate flip or rotation error.

Pass condition:

```text
generated_region_projects_to_expected_pano_location = true
```

---

### Milestone 3: Anchored Vertical Generation

Expected result:

* Backside anchor is visible only as temporary context.
* Sky/ceiling and ground/floor regions are generated before the horizontal sweep.
* Vertical regions are broadly coherent with the input scene.

Pass condition:

```text
top_bottom_are_filled = true
backside_anchor_is_not_final_content = true
```

---

### Milestone 4: Horizontal Completion

Expected result:

* The middle panorama band is completed with 8 overlapping views.
* The final panorama has no large empty regions.
* Input view remains preserved.

Pass condition:

```text
full_360_middle_band_completed = true
input_region_preserved = true
```

---

### Milestone 5: Prompting and Refinement

Expected result:

* VLM produces separate global, sky/ceiling, and ground/floor prompts.
* Optional partial denoising improves local texture quality.
* Soft blending avoids obvious refinement seams.

Pass condition:

```text
directional_prompts_are_valid = true
final_panorama_is_complete_and_viewable = true
```

---

## 10. Final Recommendation

For a faithful and cost-efficient reproduction, implement the method in the following order:

```text
1. Geometry-only equirectangular projection and perspective rendering.
2. Dummy-fill progressive update loop.
3. Real inpainting backend.
4. Anchored synthesis schedule.
5. VLM prompt generation.
6. Optional partial-denoising refinement.
```

The key implementation target is not to reproduce the paper's exact visual quality, because the inpainting model is proprietary and several implementation details are underspecified. The correct target is to reproduce the **anchored progressive panorama synthesis procedure** with the reported view counts, FoVs, prompt decomposition, and final refinement behavior.
