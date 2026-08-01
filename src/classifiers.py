from typing import Any

import numpy as np


class BaseClassifier:
    def step(self, rgbd_img: np.ndarray) -> list[dict[int, dict[str, Any]]]:
        raise NotImplementedError
