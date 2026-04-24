# Unified Reproduction Stages for `A Recipe` Sec. 3.1

## Stage 1 — Geometry and Panorama Canvas

Implement the geometric foundation.

### Tasks

- Create an equirectangular panorama canvas.
- Estimate or manually specify the input image FoV.
- Project the input perspective image into the panorama.
- Render perspective crops from the panorama.
- Maintain a known-region mask and missing-region mask.

### Expected Result

The input image can be placed correctly at the front of the panorama, and rendering the front perspective view from the panorama approximately reconstructs the input image.

---

## Stage 2 — Single-View Inpainting Round Trip

Connect the geometry pipeline with an inpainting backend.

### Tasks

- Render one incomplete perspective crop from the panorama.
- Generate its inpainting mask.
- Run the inpainting model on the masked crop.
- Project the completed crop back to the panorama.
- Update the panorama and known mask.

### Expected Result

One missing perspective view can be completed and correctly written back to the panorama without obvious yaw, pitch, flip, or mask errors.

---

## Stage 3 — Anchored Panorama Synthesis

Implement the core heuristic from `A Recipe`.

### Tasks

- Duplicate the input image to the backside as a temporary anchor.
- Generate the top / sky / ceiling region first.
- Generate the bottom / ground / floor region.
- Remove the backside anchor from the valid known mask.
- Generate the horizontal middle band using 8 rotating perspective views.

### Expected Result

The full equirectangular panorama is completed using the anchored strategy, and the backside duplicate does not remain as final content.

---

## Stage 4 — VLM Prompt Generation

Implement the paper's prompt-generation behavior.

### Tasks

- Use a VLM to analyze the input image.
- Infer whether the scene is indoor or outdoor.
- Generate:
    - a coarse global atmosphere prompt,
    - a sky or ceiling prompt,
    - a ground or floor prompt,
    - an optional negative prompt.
- Use different prompts for vertical and horizontal generation.

### Expected Result

The pipeline no longer relies on a single repeated prompt. Sky/ceiling, ground/floor, and horizontal views receive region-appropriate prompts, reducing object duplication and scene mismatch.

---

## Stage 5 — Refinement and Reproducible Packaging

Finalize the method implementation.

### Tasks

- Add optional partial-denoising refinement using the last 30% of diffusion steps.
- Blend refined regions with a soft mask.
- Preserve the original input region.
- Save all intermediate debug outputs.
- Provide a CLI or config-driven script.

### Expected Result

The method runs end-to-end from a single input image to a complete 2:1 equirectangular panorama, with intermediate outputs available for debugging.

---

# Final Stage Order

The reproduction should follow this order:

```text
Stage 1: Geometry and panorama canvas
Stage 2: Single-view inpainting round trip
Stage 3: Anchored panorama synthesis
Stage 4: VLM prompt generation
Stage 5: Refinement and packaging