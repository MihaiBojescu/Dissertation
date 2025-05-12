import random
from torch import Tensor, rand, zeros
from common.utils.noisify import BaseNoisify


class MidNoisify(BaseNoisify):
    def __call__(self, x: Tensor) -> Tensor:
        start = random.randint(a=0, b=x.shape[0])
        end = random.randint(a=start, b=x.shape[0])
        result = zeros((self._samples, x.shape[0], x.shape[1]), dtype=x.dtype)

        for i in range(0, self._samples):
            result[i, :, :] = x

        for i in range(self._samples - 2, -1, -1):
            result[i, :, start:end] = result[i + 1, :, start:end] + (
                rand((x.shape[0], end - start), dtype=x.dtype) * 0.1
            )

        return result
