import torch


class TimeEmbedding(torch.nn.Module):
    _dim: int
    _net: torch.nn.Module

    def __init__(self, dim: int):
        super().__init__()
        self._dim = dim
        self._net = torch.nn.Sequential(
            torch.nn.Linear(self._dim, self._dim * 4),
            torch.nn.Tanh(),
            torch.nn.Linear(self._dim * 4, self._dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self._dim // 2
        frequencies = torch.exp(
            -torch.log(torch.tensor(1000.0, dtype=torch.float32))
            * torch.arange(half, dtype=torch.float32)
            / half
        )
        x = x[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

        return self._net(embedding)
