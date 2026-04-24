def single_view_schedule(yaw=45.0, pitch=0.0, fov_x=85.0, fov_y=None):
    if fov_y is None:
        fov_y = fov_x

    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "fov_x": float(fov_x),
        "fov_y": float(fov_y),
    }
