import random
from torch import Tensor
from common.utils.noisify import BaseNoisify


class EndNoisify(BaseNoisify):
    def __call__(self, x: Tensor) -> Tensor:
        start = random.randint(
            a=x.shape[0] - int(x.shape[0] * self._max_percentage), b=x.shape[0]
        )
        end = x.shape[0]

        return self._noisify(x, start, end)
