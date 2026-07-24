import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_lwh(
        mask: np.ndarray,
        depth: np.ndarray,
        intrinsics: dict[str, float],
        camera_height: float,
        image_h: float,
        image_w: float,
) -> tuple[float, float, float]:
    u_grid, v_grid = np.meshgrid(np.arange(image_w), np.arange(image_h))

    Z = depth[mask].astype(float)
    U = u_grid[mask]
    V = v_grid[mask]

    if not len(Z):
        logger.warning('No object detected on the conveyor')
        return (float('nan'),) * 3

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


def get_roundness(mask: np.ndarray, verbose: bool = False) -> float | tuple[float, float, float]:
    object_mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not len(contours):
        logger.warning('No object detected on the conveyor')
        if verbose:
            return (float('nan'),) * 3
        else:
            return float('nan')

    largest_contour = max(contours, key=cv2.contourArea)
    _, r_enc_px = cv2.minEnclosingCircle(largest_contour)  # firs output is (x, y) coordinates of center of the circle

    clean_mask = np.zeros_like(object_mask)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
    dist_map = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    r_ins_px = dist_map.max()  # use cv2.minMaxLoc in center location is needed

    roundness = r_ins_px / r_enc_px
    if verbose:
        return r_ins_px, r_enc_px, roundness
    return roundness

def depth_mask(depth_img: np.ndarray, camera_height: float, height_thresh: float) -> np.ndarray:
    return (depth_img > 0) & (depth_img < (camera_height - height_thresh))


def main():
    from pathlib import Path
    import h5py
    obj = 'detergent'
    obj_id = 0
    ds_dir = Path(f'datasets/gazebo_one/{obj}')

    data = h5py.File(ds_dir / f'{obj_id}.hdf5', 'r')

    depth = np.array(data['depth']) * 1000  # mm

    im_h, im_w = depth.shape

    s_x = im_w / 640
    s_y = im_h / 480

    fx = 355.0066183 * s_x
    fy = 355.066183 * s_y
    cx = 320 * s_x
    cy = 240 * s_y

    intrinsics = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
    }

    camera_height: float = 1000  # mm
    height_thresh: float = 10  # mm

    valid_mask = depth_mask(depth, camera_height, height_thresh)

    l, w, h = get_lwh(
        mask=valid_mask,
        depth=depth,
        intrinsics=intrinsics,
        camera_height=camera_height,
        image_h=im_h,
        image_w=im_w,
    )

    # roundness
    r_ins_px, r_enc_px, roundness = get_roundness(valid_mask, verbose=True)

    print(f'Object:      {obj:} #{obj_id}')
    print(f'Length:      {l:.2f} mm')
    print(f'Width:       {w:.2f} mm')
    print(f'Height:      {h:.2f} mm')
    print(f'Roundness:  {roundness:.4f}')
    print(f'R enclosing: {r_enc_px:.2f} px')
    print(f'R inscribed: {r_ins_px:.2f} px')


if __name__ == '__main__':
    main()