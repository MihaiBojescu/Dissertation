import torch


class Transformer(torch.nn.Module):
    _dims: int
    _heads: int
    _device: torch.device
    _norm1: torch.nn.Module
    _attention: torch.nn.Module
    _norm2: torch.nn.Module
    _linear: torch.nn.Module

    def __init__(
        self, dims: int, heads: int, device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self._dims = dims
        self._heads = heads
        self._device = device

        self._norm1 = torch.nn.LayerNorm(self._dims)
        self._attention = torch.nn.MultiheadAttention(self._dims, self._heads)
        self._norm2 = torch.nn.LayerNorm(self._dims)
        self._linear = torch.nn.Sequential(
            torch.nn.Linear(self._dims, self._dims * 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self._dims * 4, self._dims),
        )

        self._norm1 = self._norm1.to(self._device)
        self._attention = self._attention.to(self._device)
        self._norm2 = self._norm2.to(self._device)
        self._linear = self._linear.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self._norm1(x)
        x = x + self._attention(x_norm, x_norm, x_norm)[0]
        x_norm = self._norm2(x)
        x = x + self._linear(x_norm)

        return x
