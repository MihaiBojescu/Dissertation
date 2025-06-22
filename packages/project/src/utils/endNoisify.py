import random
import torch
from utils.noisify import BaseNoisify


class EndNoisify(BaseNoisify):
    _max_percentage: float

    def __init__(
        self,
        samples: int,
        max_percentage: float,
        beta_start: float = 0.0025,
        beta_end: float = 0.05,
    ):
        super().__init__(samples, beta_start, beta_end)
        self._max_percentage = max_percentage

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        start = random.randint(
            a=x.shape[-1] - int(x.shape[-1] * self._max_percentage), b=x.shape[-1]
        )
        end = x.shape[-1]
        result = [(x, torch.tensor(i)) for i in range(self.samples)]

        for i in range(1, self.samples):
            previous_x = result[i - 1][0]
            current_x = self._noisify(previous_x, start, end, i)

            result[i] = (current_x, torch.tensor(i))

        return result
