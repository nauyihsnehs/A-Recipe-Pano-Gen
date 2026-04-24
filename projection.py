import math

import cv2
import numpy as np


def _rotation_matrix(yaw, pitch):
    yaw = math.radians(float(yaw))
    pitch = math.radians(float(pitch))

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    yaw_matrix = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )
    pitch_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, sp],
            [0.0, -sp, cp],
        ],
        dtype=np.float32,
    )

    return yaw_matrix @ pitch_matrix


def _check_fov(fov_x, fov_y):
    fov_x = float(fov_x)
    fov_y = float(fov_y)

    if fov_x <= 0.0 or fov_x >= 180.0:
        raise ValueError("fov_x must be between 0 and 180 degrees")
    if fov_y <= 0.0 or fov_y >= 180.0:
        raise ValueError("fov_y must be between 0 and 180 degrees")

    return fov_x, fov_y


def _parse_size(size):
    if len(size) != 2:
        raise ValueError("size must be a (width, height) pair")

    width = int(size[0])
    height = int(size[1])

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    return width, height


def _perspective_rays(width, height, fov_x, fov_y):
    fov_x, fov_y = _check_fov(fov_x, fov_y)
    tan_x = math.tan(math.radians(fov_x) * 0.5)
    tan_y = math.tan(math.radians(fov_y) * 0.5)

    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    x = (x * 2.0 - 1.0) * tan_x
    y = (y * 2.0 - 1.0) * tan_y

    xv, yv = np.meshgrid(x, y)
    rays = np.stack([xv, -yv, np.ones_like(xv)], axis=-1)
    norm = np.linalg.norm(rays, axis=-1, keepdims=True)

    return rays / norm


def _equirectangular_rays(width, height):
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height

    theta = x * (2.0 * math.pi) - math.pi
    latitude = (math.pi * 0.5) - y * math.pi

    theta, latitude = np.meshgrid(theta, latitude)
    cos_latitude = np.cos(latitude)

    rays = np.stack(
        [
            cos_latitude * np.sin(theta),
            np.sin(latitude),
            cos_latitude * np.cos(theta),
        ],
        axis=-1,
    )

    return rays.astype(np.float32)


def project_perspective_to_equirect(image, fov_x, fov_y, yaw, pitch, pano_size):
    pano_width, pano_height = _parse_size(pano_size)
    src_height, src_width = image.shape[:2]
    fov_x, fov_y = _check_fov(fov_x, fov_y)

    if pano_width != pano_height * 2:
        raise ValueError("pano_size must use a 2:1 width:height ratio")

    tan_x = math.tan(math.radians(fov_x) * 0.5)
    tan_y = math.tan(math.radians(fov_y) * 0.5)
    rotation = _rotation_matrix(yaw, pitch)

    world_rays = _equirectangular_rays(pano_width, pano_height)
    camera_rays = world_rays @ rotation
    camera_z = camera_rays[..., 2]

    x_norm = np.zeros_like(camera_z, dtype=np.float32)
    y_norm = np.zeros_like(camera_z, dtype=np.float32)
    visible = camera_z > 1e-6

    np.divide(camera_rays[..., 0], camera_z, out=x_norm, where=visible)
    np.divide(camera_rays[..., 1], camera_z, out=y_norm, where=visible)

    map_x = ((x_norm / tan_x) + 1.0) * (src_width * 0.5) - 0.5
    map_y = ((-y_norm / tan_y) + 1.0) * (src_height * 0.5) - 0.5
    valid = (
        visible
        & (np.abs(x_norm) <= tan_x)
        & (np.abs(y_norm) <= tan_y)
        & (map_x >= 0.0)
        & (map_x <= src_width - 1)
        & (map_y >= 0.0)
        & (map_y <= src_height - 1)
    )

    map_x = np.where(valid, map_x, 0.0).astype(np.float32)
    map_y = np.where(valid, map_y, 0.0).astype(np.float32)

    projected = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    projection_mask = np.where(valid, 255, 0).astype(np.uint8)

    projected[projection_mask == 0] = 0

    return projected, projection_mask


def render_perspective_from_equirect(panorama, yaw, pitch, fov_x, fov_y, out_size):
    out_width, out_height = _parse_size(out_size)
    pano_height, pano_width = panorama.shape[:2]
    fov_x, fov_y = _check_fov(fov_x, fov_y)

    rays = _perspective_rays(out_width, out_height, fov_x, fov_y)
    rotation = _rotation_matrix(yaw, pitch)
    world_rays = rays @ rotation.T

    longitude = np.arctan2(world_rays[..., 0], world_rays[..., 2])
    latitude = np.arcsin(np.clip(world_rays[..., 1], -1.0, 1.0))

    map_x = ((longitude + math.pi) / (2.0 * math.pi)) * pano_width - 0.5
    map_y = ((math.pi * 0.5 - latitude) / math.pi) * pano_height - 0.5
    map_x = np.mod(map_x, pano_width).astype(np.float32)
    map_y = np.clip(map_y, 0.0, pano_height - 1).astype(np.float32)

    return cv2.remap(
        panorama,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def render_perspective_mask_from_equirect(mask, yaw, pitch, fov_x, fov_y, out_size):
    out_width, out_height = _parse_size(out_size)
    pano_height, pano_width = mask.shape[:2]
    fov_x, fov_y = _check_fov(fov_x, fov_y)

    rays = _perspective_rays(out_width, out_height, fov_x, fov_y)
    rotation = _rotation_matrix(yaw, pitch)
    world_rays = rays @ rotation.T

    longitude = np.arctan2(world_rays[..., 0], world_rays[..., 2])
    latitude = np.arcsin(np.clip(world_rays[..., 1], -1.0, 1.0))

    map_x = ((longitude + math.pi) / (2.0 * math.pi)) * pano_width - 0.5
    map_y = ((math.pi * 0.5 - latitude) / math.pi) * pano_height - 0.5
    map_x = np.mod(map_x, pano_width).astype(np.float32)
    map_y = np.clip(map_y, 0.0, pano_height - 1).astype(np.float32)

    return cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_WRAP,
    )


project_input_to_equirectangular = project_perspective_to_equirect
render_perspective_from_equirectangular = render_perspective_from_equirect
