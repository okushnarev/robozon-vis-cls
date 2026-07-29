import numpy as np


def unproject_depth(
        depth: np.ndarray,
        intrinsics: dict[str, float],
        mask: np.ndarray| None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_h, image_w = depth.shape
    u_grid, v_grid = np.meshgrid(np.arange(image_w), np.arange(image_h))

    if mask is None:
        Z = depth.ravel().astype(float)
        U = u_grid.ravel()
        V = v_grid.ravel()
    else:
        Z = depth[mask].astype(float)
        U = u_grid[mask]
        V = v_grid[mask]

    # Apply the mathematical pinhole equations to find X and Y in millimeters
    X = (U - intrinsics['cx']) * Z / intrinsics['fx']
    Y = (V - intrinsics['cy']) * Z / intrinsics['fy']

    return X, Y, Z
