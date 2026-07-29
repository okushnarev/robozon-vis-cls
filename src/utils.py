from typing import Any

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


def get_valid_instances(
        instance_attr_maps: list[dict[str, Any]],
        instance_segmaps: np.ndarray,
        pad: int,
) -> list[dict[str, Any]]:
    """
    Removes indices of objects that are too close to left and right border from `instance_attr_maps`
    :param instance_attr_maps: Instance attribute maps. List of dictionaries that describe found classes and their unique id
    :param instance_segmaps: Map with classes' ids of dims (H, W)
    :param pad: Padding in pixels/columns from left and right border. Should be > 0
    :return: Updated valid Instance attribute maps that contain objects not touching the border
    """

    if pad <= 0:
        raise ValueError('pad must be > 0')

    l_pad = pad - 1
    r_pad = pad

    idx = np.unique(instance_segmaps[:, [l_pad, -r_pad]])

    return [d for d in instance_attr_maps if d['idx'] not in idx]
