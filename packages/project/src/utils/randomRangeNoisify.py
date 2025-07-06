import random
import torch
from utils.noisify import BaseNoisify


class RandomRangeNoisify(BaseNoisify):
    _index_chance: float

    def __init__(
        self,
        samples: int,
        beta_start: float = 0.0025,
        beta_end: float = 0.05,
    ):
        super().__init__(samples, beta_start, beta_end)

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        The indexes will be generated:
        - for start: between 0           and 3/4 of length;
        - for end:   between (start + 1) and 1/1 of length;

        Example: [----------]
                 ^----------- min start
                         ^--- max start
                         ^--- min end
                            ^ max end
        """
        start = random.randint(0, x.shape[-1] - (x.shape[-1] // 4) - 1)
        end = random.randint(start + 1, x.shape[-1] - 1)
        result = [(x, torch.tensor(i)) for i in range(self.samples)]

        for i in range(1, self.samples):
            previous_x = result[i - 1][0]
            current_x = result[i][0]
            current_x = self._noisify(previous_x, start, end, i)

            result[i] = (current_x, torch.tensor(i))

        return result
