from dataclasses import dataclass

import numpy as np
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
