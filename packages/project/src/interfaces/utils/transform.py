from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import torch

T = TypeVar("T")


class MultiDiffusionTransform(torch.nn.Module, Generic[T], ABC):
    samples: int

    def __init__(self, samples: int) -> None:
        super().__init__()
        self.samples = samples

    @abstractmethod
    def forward(self, x: torch.Tensor) -> T:
        pass
