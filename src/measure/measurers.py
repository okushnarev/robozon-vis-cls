import numpy as np

class BaseMeasurer:
    def measure(self, img: np.ndarray) -> tuple[float, ...]:
        raise NotImplementedError
