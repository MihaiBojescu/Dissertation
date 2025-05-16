import random
import torch
from common.utils.noisify import BaseNoisify


class RandomNoisify(BaseNoisify):
    _index_chance: float

    def __init__(self, samples: int, index_chance: float = 0.25):
        super().__init__(samples)
        self._index_chance = index_chance

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        picked_indices = [
            i for i in range(x.shape[-1]) if random.uniform(0, 1) > self._index_chance
        ]
        result = torch.zeros((self._samples, x.shape[0], x.shape[1]), dtype=x.dtype)

        for i in range(0, self._samples):
            result[i, :, :] = x

        for i in range(self._samples - 2, -1, -1):
            for j in picked_indices:
                start = j
                end = j + 1
                result[i, :, start:end] = self._noisify(result[i + 1], start, end)

        return result
