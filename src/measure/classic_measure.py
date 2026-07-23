import numpy as np

def depth_mask(depth_img: np.ndarray, camera_height: float, height_thresh: float) -> np.ndarray:
    return (depth_img > 0) & (depth_img < (camera_height - height_thresh))