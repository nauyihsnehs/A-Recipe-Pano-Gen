def single_view_schedule(yaw=45.0, pitch=0.0, fov_x=85.0, fov_y=None):
    if fov_y is None:
        fov_y = fov_x

    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "fov_x": float(fov_x),
        "fov_y": float(fov_y),
    }


def anchored_view_schedule(middle_fov=85.0, vertical_fov=120.0):
    views = []

    for yaw in [0, 90, 180, 270]:
        views.append(
            {
                "phase": "top",
                "yaw": float(yaw),
                "pitch": 90.0,
                "fov_x": float(vertical_fov),
                "fov_y": float(vertical_fov),
            }
        )

    for yaw in [0, 90, 180, 270]:
        views.append(
            {
                "phase": "bottom",
                "yaw": float(yaw),
                "pitch": -90.0,
                "fov_x": float(vertical_fov),
                "fov_y": float(vertical_fov),
            }
        )

    for yaw in [0, 45, 90, 135, 180, 225, 270, 315]:
        views.append(
            {
                "phase": "horizontal",
                "yaw": float(yaw),
                "pitch": 0.0,
                "fov_x": float(middle_fov),
                "fov_y": float(middle_fov),
            }
        )

    return views
