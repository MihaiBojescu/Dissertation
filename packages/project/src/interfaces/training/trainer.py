import torch
import typing as t

T = t.TypeVar("T")
U = t.TypeVar("U")

class BaseTrainer(t.Generic[T, U]):
    def train(
        self,
        dataloader: torch.utils.data.DataLoader[T],
        epochs: int,
        callback: t.Callable[[U, float], None] = lambda y_hat, loss: None,
    ) -> None:
        pass

    def eval(
        self,
        dataloader: torch.utils.data.DataLoader[T],
        callback: t.Callable[[U, float], None] = lambda y_hat, loss: None,
    ) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
