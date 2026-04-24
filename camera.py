import math


DEFAULT_FOV_X = 70.0


def estimate_or_load_fov(image, fov_x=None):
    if fov_x is None:
        fov_x = DEFAULT_FOV_X

    height, width = image.shape[:2]
    fov_x = float(fov_x)

    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if fov_x <= 0.0 or fov_x >= 180.0:
        raise ValueError("fov_x must be between 0 and 180 degrees")

    fov_x_rad = math.radians(fov_x)
    focal = (width * 0.5) / math.tan(fov_x_rad * 0.5)
    fov_y_rad = 2.0 * math.atan((height * 0.5) / focal)

    return fov_x, math.degrees(fov_y_rad)
