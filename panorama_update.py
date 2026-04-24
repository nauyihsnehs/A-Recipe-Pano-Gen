import numpy as np

from projection import project_perspective_mask_to_equirect, project_perspective_to_equirect


def update_pano_with_view(panorama, known_mask, generated_view, update_mask, yaw, pitch, fov_x, fov_y):
    if panorama.shape[:2] != known_mask.shape:
        raise ValueError("panorama and known_mask dimensions do not match")
    if generated_view.shape[:2] != update_mask.shape:
        raise ValueError("generated_view and update_mask dimensions do not match")

    pano_size = (panorama.shape[1], panorama.shape[0])
    projected_view, view_footprint = project_perspective_to_equirect(
        generated_view,
        fov_x,
        fov_y,
        yaw,
        pitch,
        pano_size,
    )
    projected_mask = project_perspective_mask_to_equirect(
        update_mask,
        fov_x,
        fov_y,
        yaw,
        pitch,
        pano_size,
    )
    projected_update_mask = np.where(
        (projected_mask > 0) & (view_footprint > 0) & (known_mask == 0),
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

    return updated_panorama, updated_known_mask, projected_update, projected_update_mask
