import random
import torch
from common.utils.noisify import BaseNoisify


class RandomNoisify(BaseNoisify):
    _index_chance: float

    def __init__(self, samples: int, index_chance: float = 0.25):
        super().__init__(samples)
        self._index_chance = index_chance

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        picked_indices = [
            i for i in range(x.shape[-1]) if random.uniform(0, 1) > self._index_chance
        ]
        result = [(x, torch.tensor(i)) for i in range(self._samples)]

        for i in range(self._samples - 2, -1, -1):
            current_x = result[i][0]
            next_x = result[i + 1][0]

            for j in picked_indices:
                start = j
                end = j + 1
                current_x = self._noisify(next_x, start, end)

            result[i] = (current_x, torch.tensor(i))

        return result
