from abc import ABC, abstractmethod
import torch


class BaseNoisify(torch.nn.Module, ABC):
    _samples: int

    def __init__(self, samples: int):
        self._samples = samples

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass
