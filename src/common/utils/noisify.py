import torch


class BaseNoisify(torch.nn.Module):
    _samples: int
    _max_percentage: float

    def __init__(self, samples: int, max_percentage: float):
        super().__init__()

        self._samples = samples
        self._max_percentage = max_percentage

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _noisify(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        result = torch.zeros((self._samples, x.shape[0], x.shape[1]), dtype=x.dtype)

        for i in range(0, self._samples):
            result[i, :, :] = x

        for i in range(self._samples - 2, -1, -1):
            result[i, :, start:end] = result[i + 1, :, start:end] + (
                torch.rand((x.shape[0], end - start), dtype=x.dtype) * 0.1
            )

        return result
