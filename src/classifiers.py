from dataclasses import dataclass

import numpy as np
from rfdetr import RFDETR
from trackers import ByteTrackTracker

from src.measure.measurers import BaseMeasurer


@dataclass
class ReturnData:
    length: float
    width: float
    height: float
    roundness: float
    is_round: bool
    center_xy: tuple[float, float]
    class_name: str


class BaseClassifier:
    def step(self, rgbd_img: np.ndarray) -> dict[int, ReturnData]:
        raise NotImplementedError


class RFDETRClassifier(BaseClassifier):
    def __init__(
            self,
            model: RFDETR,
            detection_threshold: float,
            roundness_threshold: float,
            measurer: BaseMeasurer,
            margin_px: int,
            expand_px: int,
            fps: int,
            lost_track_buffer: int = 15,
    ):
        self.model = model

        self.detection_threshold = detection_threshold
        self.roundness_threshold = roundness_threshold
        self.measurer = measurer
        self.margin_px = margin_px
        self.expand_px = expand_px

        self.fps = fps
        self.lost_track_buffer = lost_track_buffer

        self.tracker = ByteTrackTracker(
            frame_rate=self.fps,
            lost_track_buffer=self.lost_track_buffer,
        )

        self.item_params: dict[int, ReturnData] = {}
        self.img_shape_xy: np.ndarray = None
        self.img_center: np.ndarray = None

    def _check_image_params_unset(self, rgb_img: np.ndarray) -> None:
        if self.img_shape_xy is None:
            self.img_shape_xy = np.array(rgb_img.shape[1::-1])
            self.img_center = self.img_shape_xy / 2

    def _prep_return(self, visible_classes: list[int]) -> dict[int, ReturnData]:
        return {idx: self.item_params[idx] for idx in visible_classes if idx >= 0 and idx in self.item_params}

    def step(self, rgbd_img: np.ndarray) -> dict[int, ReturnData]:
        colors = rgbd_img[:, :, :3].astype(np.uint8)
        depth = rgbd_img[:, :, 3]
        self._check_image_params_unset(colors)

        detections = self.model.predict(colors, threshold=self.detection_threshold)
        detections = self.tracker.update(detections=detections)

        for idx, det in enumerate(detections):
            det_idx = det[4]
            det_class = det[5]['class_name']

            lt_corner = det[0][:2].copy()
            rb_corner = det[0][2:].copy()
            bbox_len = rb_corner - lt_corner
            bbox_center = lt_corner + (bbox_len) / 2

            if det_idx in self.item_params:
                self.item_params[det_idx].center_xy = bbox_center.tolist()
            else:
                is_in_bounds = abs(self.img_center[0] - bbox_center[0]) < self.margin_px
                if is_in_bounds:
                    expanded_bbox_lt = (lt_corner - self.expand_px).astype(np.intp)
                    expanded_bbox_rb = (rb_corner + self.expand_px).astype(np.intp)

                    expanded_bbox_lt = np.clip(expanded_bbox_lt, (0, 0), self.img_shape_xy)
                    expanded_bbox_rb = np.clip(expanded_bbox_rb, (0, 0), self.img_shape_xy)

                    detections.xyxy[idx] = np.array([*expanded_bbox_lt, *expanded_bbox_rb])

                    slice_idx = (
                        slice(expanded_bbox_lt[1], expanded_bbox_rb[1] + 1),
                        slice(expanded_bbox_lt[0], expanded_bbox_rb[0] + 1)
                    )
                    l, w, h, roundness = self.measurer.measure(depth[slice_idx])

                    is_round = roundness > self.roundness_threshold

                    self.item_params[det_idx] = ReturnData(
                        length=l,
                        width=w,
                        height=h,
                        roundness=roundness,
                        is_round=is_round,
                        center_xy=bbox_center.tolist(),
                        class_name=det_class,
                    )

        return self._prep_return(detections.tracker_id.tolist())
