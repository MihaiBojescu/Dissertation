import random
from torch import Tensor
from common.utils.noisify import BaseNoisify


class MidNoisify(BaseNoisify):
    def __call__(self, x: Tensor) -> Tensor:
        start = random.randint(
            a=0, b=x.shape[0] - int(x.shape[0] * self._max_percentage)
        )
        end = random.randint(
            a=start,
            b=max(start + int(x.shape[0] * self._max_percentage), x.shape[0]),
        )

        return self._noisify(x, start, end)
