import random
import torch
from common.utils.noisify import BaseNoisify


class StartNoisify(BaseNoisify):
    _max_percentage: float

    def __init__(self, samples: int, max_percentage: float):
        super().__init__(samples)
        self._max_percentage = max_percentage

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        start = 0
        end = random.randint(a=start, b=int(x.shape[0] * self._max_percentage))
        result = [(x, torch.tensor(i)) for i in range(self._samples)]

        for i in range(self._samples - 2, -1, -1):
            current_x = result[i][0]
            next_x = result[i + 1][0]

            current_x = self._noisify(next_x, start, end)

            result[i] = (current_x, torch.tensor(i))

        return result

