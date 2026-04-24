import numpy as np

from canvas import create_equirectangular_canvas, paste_projected_view
from inpaint_runner import run_inpainting
from mask import compute_inpaint_mask
from panorama_update import update_pano_with_view
from projection import (
    project_perspective_to_equirect,
    render_perspective_from_equirect,
    render_perspective_mask_from_equirect,
)
from view_scheduler import anchored_view_schedule


def initialize_anchored_panorama(input_image, input_fov_x, input_fov_y, pano_size):
    panorama, known_mask = create_equirectangular_canvas(pano_size[0], pano_size[1])
    front_image, input_mask = project_perspective_to_equirect(
        input_image,
        input_fov_x,
        input_fov_y,
        0.0,
        0.0,
        pano_size,
    )
    panorama, known_mask = paste_projected_view(
        panorama,
        known_mask,
        front_image,
        input_mask,
    )

    anchor_image, anchor_mask = project_perspective_to_equirect(
        input_image,
        input_fov_x,
        input_fov_y,
        180.0,
        0.0,
        pano_size,
    )
    panorama, known_mask = paste_projected_view(
        panorama,
        known_mask,
        anchor_image,
        anchor_mask,
    )

    input_mask = np.where(input_mask > 0, 255, 0).astype(np.uint8)
    anchor_mask = np.where(anchor_mask > 0, 255, 0).astype(np.uint8)

    return panorama, known_mask, input_mask, anchor_mask


def remove_backside_anchor(panorama, known_mask, input_mask, anchor_mask):
    output = panorama.copy()
    output_mask = known_mask.copy()
    anchor_only = (anchor_mask > 0) & (input_mask == 0)

    output[anchor_only] = 0
    output_mask[anchor_only] = 0
    output_mask[input_mask > 0] = 255

    return output, output_mask


def make_masked_view(view, mask):
    masked = view.copy()
    masked[mask > 0] = 0

    return masked


def prompt_for_view(view, global_prompt, top_prompt, bottom_prompt):
    phase = view["phase"]

    if phase == "top":
        return combine_prompts(top_prompt, global_prompt)
    if phase == "bottom":
        return combine_prompts(bottom_prompt, global_prompt)

    return global_prompt


def combine_prompts(primary, secondary):
    primary = primary.strip()
    secondary = secondary.strip()

    if primary and secondary:
        return primary + ", " + secondary
    if primary:
        return primary

    return secondary


def run_anchored_step(
    panorama,
    known_mask,
    view,
    backend,
    prompt,
    negative_prompt,
    seed,
    num_steps,
    guidance_scale,
    view_size,
):
    out_size = (view_size, view_size)
    rendered_view = render_perspective_from_equirect(
        panorama,
        view["yaw"],
        view["pitch"],
        view["fov_x"],
        view["fov_y"],
        out_size,
    )
    view_known_mask = render_perspective_mask_from_equirect(
        known_mask,
        view["yaw"],
        view["pitch"],
        view["fov_x"],
        view["fov_y"],
        out_size,
    )
    inpaint_mask = compute_inpaint_mask(view_known_mask)
    masked_view = make_masked_view(rendered_view, inpaint_mask)
    inpainted_view = run_inpainting(
        masked_view,
        inpaint_mask,
        prompt,
        negative_prompt,
        seed,
        backend,
        num_steps,
        guidance_scale,
    )
    updated_panorama, updated_known_mask, projected_update, projected_update_mask = update_pano_with_view(
        panorama,
        known_mask,
        inpainted_view,
        inpaint_mask,
        view["yaw"],
        view["pitch"],
        view["fov_x"],
        view["fov_y"],
    )

    debug_images = {
        "rendered_view": rendered_view,
        "view_known_mask": view_known_mask,
        "inpaint_mask": inpaint_mask,
        "masked_view": masked_view,
        "inpainted_view": inpainted_view,
        "projected_update": projected_update,
        "projected_update_mask": projected_update_mask,
        "updated_panorama": updated_panorama,
        "updated_known_mask": updated_known_mask,
    }

    return updated_panorama, updated_known_mask, debug_images


def run_anchored_synthesis(
    input_image,
    input_fov_x,
    input_fov_y,
    pano_size,
    backend,
    global_prompt,
    top_prompt,
    bottom_prompt,
    negative_prompt,
    seed=42,
    num_steps=40,
    guidance_scale=7.5,
    view_size=1024,
    middle_fov=85.0,
    vertical_fov=120.0,
    debug_writer=None,
):
    panorama, known_mask, input_mask, anchor_mask = initialize_anchored_panorama(
        input_image,
        input_fov_x,
        input_fov_y,
        pano_size,
    )
    records = []
    anchor_removed = False
    schedule = anchored_view_schedule(middle_fov=middle_fov, vertical_fov=vertical_fov)

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
        if view["phase"] == "horizontal" and not anchor_removed:
            if debug_writer:
                debug_writer(
                    "after_vertical",
                    {
                        "panorama": panorama,
                        "known_mask": known_mask,
                    },
                )

            panorama, known_mask = remove_backside_anchor(
                panorama,
                known_mask,
                input_mask,
                anchor_mask,
            )
            anchor_removed = True

            if debug_writer:
                debug_writer(
                    "anchor_removed",
                    {
                        "panorama": panorama,
                        "known_mask": known_mask,
                    },
                )

        known_before = int(np.count_nonzero(known_mask))
        prompt = prompt_for_view(view, global_prompt, top_prompt, bottom_prompt)
        panorama, known_mask, debug_images = run_anchored_step(
            panorama,
            known_mask,
            view,
            backend,
            prompt,
            negative_prompt,
            int(seed) + index,
            num_steps,
            guidance_scale,
            view_size,
        )
        known_after = int(np.count_nonzero(known_mask))
        record = {
            "index": index,
            "phase": view["phase"],
            "yaw": view["yaw"],
            "pitch": view["pitch"],
            "fov_x": view["fov_x"],
            "fov_y": view["fov_y"],
            "known_before": known_before,
            "known_after": known_after,
            "known_added": known_after - known_before,
            "prompt": prompt,
        }
        records.append(record)

        if debug_writer:
            payload = debug_images.copy()
            payload["record"] = record
            debug_writer("step", payload)

    if debug_writer:
        debug_writer(
            "final",
            {
                "panorama": panorama,
                "known_mask": known_mask,
            },
        )

    return panorama, known_mask, records
