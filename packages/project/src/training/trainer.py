import typing as t
import torch
import tqdm
from interfaces.training.trainer import BaseTrainer


class ModelTrainer(
    BaseTrainer[t.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
):
    _model: torch.nn.Module
    _optimiser: torch.optim.Optimizer
    _loss_function: torch.nn.Module

    def __init__(
        self,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        loss_function: torch.nn.Module,
    ):
        self._model = model
        self._optimiser = optimiser
        self._loss_function = loss_function

    def train(
        self,
        dataloader: torch.utils.data.DataLoader[
            t.Tuple[torch.Tensor, torch.Tensor]
        ],
        epochs: int,
        callback: t.Callable[[torch.Tensor, float], None] = lambda y_hat, loss: None,
    ):
        self._model.train()

        for _ in tqdm.tqdm(range(epochs), position=0, unit="iter"):
            for x, y, step in tqdm.tqdm(
                dataloader, position=1, leave=False, unit="batch"
            ):
                x = x.to(torch.get_default_device())
                y = y.to(torch.get_default_device())
                step = step.to(torch.get_default_device())

                self._optimiser.zero_grad()
                y_hat = self._model(x=x, step=step)
                loss = self._loss_function(y_hat, y)
                loss.backward()
                self._optimiser.step()

                callback(y_hat, loss.item())

    def eval(
        self,
        dataloader: torch.utils.data.DataLoader[
            t.Tuple[torch.Tensor, torch.Tensor]
        ],
        callback: t.Callable[
            [torch.Tensor, float], None
        ] = lambda y_hat, loss: None,
    ):
        self._model.eval()

        for x, y, step in tqdm.tqdm(
            dataloader, position=1, leave=False, unit="batch"
        ):
            x = x.to(torch.get_default_device())
            y = y.to(torch.get_default_device())
            step = step.to(torch.get_default_device())

            self._optimiser.zero_grad()
            y_hat = self._model(x=x, step=step)
            loss = y_hat.loss

            callback(y_hat, loss.item())

    def save(self, path: str):
        torch.save(self._model.state_dict(), path)

    def load(self, path: str):
        self._model.load_state_dict(
            torch.load(path, weights_only=True, map_location=torch.get_default_device())
        )
        self._model.eval()
