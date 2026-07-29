import numpy as np


def unproject_depth(
        depth: np.ndarray,
        intrinsics: dict[str, float],
        mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts depth pixels to 3D coordinates in the camera frame
    :param depth: Depth map (H, W) in meters
    :param intrinsics: Dict with camera params 'fx', 'fy', 'cx', 'cy' in pixels
    :param mask: (default: None) Boolean array (H, W) selecting which pixels to unproject.
        If None, unprojects whole image
    :return: Tuple of (X, Y, Z) 1D arrays in meter
    """
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


def compute_grid_mapping(
        depth: np.ndarray,
        intrinsics: dict[str, float],
        mask: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Computes the 2D grid mapping indices for 3D unprojected points
    Yields result similar to orthographic projection

    :param depth: Depth map of shape (H, W)
    :param intrinsics: Camera intrinsics dictionary
    :param mask: Boolean mask filtering active pixels, defaults to None
    :param rotation: Rotation matrix of shape (3, 3), defaults to None

    :return: A tuple containing (row_indices, col_indices, sorted_point_indices, (grid_height, grid_width))
    """
    X, Y, Z = unproject_depth(depth, intrinsics, mask)

    if X.size == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp), np.array([], dtype=np.intp), (0, 0)

    if rotation is not None:
        points_3d = np.stack((X, Y, Z), axis=0)
        points_3d = rotation @ points_3d
        X, Y, Z = points_3d[0], points_3d[1], points_3d[2]

    # Pixel sizes in meters
    px_sz_x = 1.0 / intrinsics['fx']
    px_sz_y = 1.0 / intrinsics['fy']

    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    # Determine size of projection grid
    # Ensure grid resolution is at least 1x1
    w = max(1, int(np.ceil((x_max - x_min) / px_sz_x)))
    h = max(1, int(np.ceil((y_max - y_min) / px_sz_y)))

    row_idx = np.clip(((Y - y_min) / px_sz_y).round().astype(np.intp), 0, h - 1)
    col_idx = np.clip(((X - x_min) / px_sz_x).round().astype(np.intp), 0, w - 1)

    # Sort in descending order of Z (farthest first)
    # Closer points will overwrite farther points when assigned to the grid
    sort_indices = np.argsort(Z, descending=True)

    return row_idx[sort_indices], col_idx[sort_indices], sort_indices, (h, w)


def rasterize_depth(
        depth: np.ndarray,
        mapping: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
        mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterizes depth values into a 2D grid using a precomputed spatial mapping.

    :param depth: Depth map of shape (H, W) containing depth values
    :param mapping: Precomputed spatial mapping containing row indices, column indices,
                    sorted indices, and grid dimensions (h, w)
    :param mask: Boolean mask matching the original depth map dimensions, defaults to None

    :return: Projected 2D depth map of shape (h, w)
    """
    rows, cols, sort_idx, (h, w) = mapping
    if h == 0 or w == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Extract active depth values
    Z = depth[mask].astype(np.float32) if mask is not None else depth.ravel().astype(np.float32)

    depth_img = np.zeros((h, w), dtype=np.float32)
    depth_img[rows, cols] = Z[sort_idx]
    return depth_img


def rasterize_colors(
        colors: np.ndarray,
        mapping: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
        mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterizes color values into a 2D grid using a precomputed spatial mapping

    :param colors: Color array of shape (H, W, 3) or (N, 3)
    :param mapping: Precomputed spatial mapping containing row indices, column indices,
                    sorted indices, and grid dimensions (h, w)
    :param mask: Boolean mask matching the original colors/depth dimensions, defaults to None

    :return: Projected 2D color image of shape (h, w, 3)
    """
    rows, cols, sort_idx, (h, w) = mapping
    if h == 0 or w == 0:
        return np.zeros((0, 0, 3), dtype=colors.dtype)

    # Extract active colors
    colors_flat = colors[mask] if mask is not None else colors.reshape(-1, 3)

    color_img = np.zeros((h, w, 3), dtype=colors.dtype)
    color_img[rows, cols] = colors_flat[sort_idx]
    return color_img