import random
import torch
from utils.noisify import BaseNoisify


class RandomNoisify(BaseNoisify):
    _index_chance: float

    def __init__(
        self,
        samples: int,
        index_chance: float = 0.25,
        beta_start: float = 0.0025,
        beta_end: float = 0.05,
    ):
        super().__init__(samples, beta_start, beta_end)
        self._index_chance = index_chance

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        picked_indices = [
            i for i in range(x.shape[-1]) if random.uniform(0, 1) < self._index_chance
        ]
        result = [(x, torch.tensor(i)) for i in range(self.samples)]

        for i in range(1, self.samples):
            previous_x = result[i - 1][0]
            current_x = result[i][0]

            for j in picked_indices:
                start = j
                end = j + 1
                current_x = self._noisify(previous_x, start, end, i)

            result[i] = (current_x, torch.tensor(i))

        return result
