import torch


class BaseNoisify(torch.nn.Module):
    _samples: int

    def __init__(self, samples: int):
        super().__init__()
        self._samples = samples

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _noisify(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return x[:, start:end] + (
            torch.rand((x.shape[0], end - start), dtype=x.dtype) * 0.1
        )
