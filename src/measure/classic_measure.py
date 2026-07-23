import cv2
import numpy as np


def get_lwh(
        mask: np.ndarray,
        depth: np.ndarray,
        intrinsics: list[list[float]],
        camera_height: float,
        image_h: float,
        image_w: float,
) -> tuple[float, float, float] | None:
    u_grid, v_grid = np.meshgrid(np.arange(image_w), np.arange(image_h))

    Z = depth[mask].astype(float)
    U = u_grid[mask]
    V = v_grid[mask]

    if not len(Z):
        print('No object detected on the conveyor')
        return None

    # Apply the mathematical pinhole equations to find X and Y in millimeters
    X = (U - intrinsics['cx']) * Z / intrinsics['fx']
    Y = (V - intrinsics['cy']) * Z / intrinsics['fy']

    # Measurements
    height_mm = float(camera_height - np.min(Z))

    points_2d = np.stack((X, Y), axis=-1).astype(np.float32)
    rect = cv2.minAreaRect(points_2d)
    dim1, dim2 = rect[1]
    length_mm = max(dim1, dim2)
    width_mm = min(dim1, dim2)

    return length_mm, width_mm, height_mm
def depth_mask(depth_img: np.ndarray, camera_height: float, height_thresh: float) -> np.ndarray:
    return (depth_img > 0) & (depth_img < (camera_height - height_thresh))