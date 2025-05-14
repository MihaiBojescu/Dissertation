import typing as t
import torch
import tqdm
from torch.optim import AdamW
from common.utils.trainer import BaseTrainer


class ModelTrainer(
    BaseTrainer[t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
):
    _model: torch.nn.Module

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 5e-5,
    ):
        self._model = model
        self._optimizer = AdamW(model.parameters(), lr=lr)

    def train(
        self,
        dataloader: torch.utils.data.DataLoader[
            t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        epochs: int,
        callback: t.Callable[[torch.Tensor, float], None] = lambda y_hat, loss: None,
    ):
        self._model.train()

        for _ in tqdm.tqdm(range(epochs), position=0, unit="iters"):
            for x, attention_mask, y in tqdm.tqdm(
                dataloader, position=1, leave=False, unit="batches"
            ):
                x = x.to(torch.get_default_device())
                attention_mask = attention_mask.to(torch.get_default_device())
                y = y.to(torch.get_default_device())

                self._optimizer.zero_grad()
                y_hat = self._model(x, attention_mask=attention_mask, labels=y)
                loss = y_hat.loss
                loss.backward()
                self._optimizer.step()

                callback(y_hat, loss.item())

    def eval(
        self,
        dataloader: torch.utils.data.DataLoader[
            t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        callback: t.Callable[
            [torch.Tensor, float, float, float], None
        ] = lambda y_hat, loss, accuracy, f1_score: None,
    ):
        self._model.eval()

        accuracy = 0.0
        accuracy_entries = 0
        f1_scorer: None | F1Score = None

        for x, attention_mask, y in tqdm.tqdm(
            dataloader, position=1, leave=False, unit="batches"
        ):
            x = x.to(torch.get_default_device())
            attention_mask = attention_mask.to(torch.get_default_device())
            y = y.to(torch.get_default_device())

            self._optimizer.zero_grad()
            y_hat = self._model(x=x, attention_mask=attention_mask, labels=y)
            loss = y_hat.loss

            logits = y_hat.logits
            preds = torch.argmax(logits, dim=1)
            targets = torch.argmax(y, dim=1)

            accuracy *= accuracy_entries
            accuracy += (preds == targets).sum().item() / y.shape[0]
            accuracy /= accuracy_entries + 1
            accuracy_entries += 1

            if not f1_scorer:
                f1_scorer = F1Score(
                    task="multiclass", num_classes=y.shape[1], average="weighted"
                )

            f1_scorer.update(preds, targets)
            f1_score: float = f1_scorer.compute().item()

            callback(y_hat, loss.item(), accuracy, f1_score)

    def save(self, path: str):
        torch.save(self._model.state_dict(), path)

    def load(self, path: str):
        self._model.load_state_dict(
            torch.load(path, weights_only=True, map_location=torch.get_default_device())
        )
        self._model.eval()
