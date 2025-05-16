import torch


class BaseNoisify(torch.nn.Module):
    _samples: int

    def __init__(self, samples: int):
        super().__init__()
        self._samples = samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _noisify(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        x = x.clone()
        x[..., start:end] = x[..., start:end] + (
            torch.rand((x.shape[-2], end - start), dtype=x.dtype) * 0.1
        )
        return x
