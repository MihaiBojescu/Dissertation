import random
from torch import Tensor
from common.utils.noisify import BaseNoisify


class StartNoisify(BaseNoisify):
    def __call__(self, x: Tensor) -> Tensor:
        start = 0
        end = random.randint(a=start, b=int(x.shape[0] * self._max_percentage))

        return self._noisify(x, start, end)
