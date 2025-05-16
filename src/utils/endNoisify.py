import random
import torch
from common.utils.noisify import BaseNoisify


class EndNoisify(BaseNoisify):
    _max_percentage: float

    def __init__(self, samples: int, max_percentage: float):
        super().__init__(samples)
        self._max_percentage = max_percentage

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        start = random.randint(
            a=x.shape[0] - int(x.shape[0] * self._max_percentage), b=x.shape[0]
        )
        end = x.shape[0]
        result = torch.zeros((self._samples, x.shape[0], x.shape[1]), dtype=x.dtype)

        for i in range(0, self._samples):
            result[i, :, :] = x

        for i in range(self._samples - 2, -1, -1):
            result[i, :, start:end] = self._noisify(result[i + 1], start, end)

        return result

