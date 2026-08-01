import cv2
import numpy as np

from src.measure.classic_measure import depth_mask, get_lwh, get_roundness
from src.projection import compute_grid_mapping, rasterize_depth


class BaseMeasurer:
    def measure(self, img: np.ndarray) -> tuple[float, ...]:
        raise NotImplementedError


class ClassicMeasurer(BaseMeasurer):
    def __init__(
            self,
            camera_height: float,
            height_thresh: float,
            intrinsics: dict[str, float],
            kernel_size: tuple[int, int] = (5, 5),
    ):
        self.camera_height = camera_height
        self.height_thresh = height_thresh
        self.intrinsics = intrinsics
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    def measure(self, img: np.ndarray) -> tuple[float, ...]:
        mask = depth_mask(
            img,
            camera_height=self.camera_height,
            height_thresh=self.height_thresh,
        )

        l, w, h = get_lwh(
            mask=mask,
            depth=img,
            intrinsics=self.intrinsics,
            camera_height=self.camera_height,
        )
        mapping = compute_grid_mapping(
            img, self.intrinsics, mask
        )
        unproj_depth_img = rasterize_depth(
            img, mapping, mask
        )
        bin_img = (unproj_depth_img > 0).astype(np.float32)
        closing_img = cv2.morphologyEx(
            bin_img,
            cv2.MORPH_CLOSE,
            self.kernel,
        )
        roundness = get_roundness(closing_img)

        return l, w, h, roundness
