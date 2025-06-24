import torch


class PatchDeEmbedding(torch.nn.Module):
    _dims: int
    _channels: int
    _patch_size: tuple[int, int]
    _net: torch.nn.Module

    def __init__(self, dims: int, channels: int, patch_size: tuple[int, int]):
        super().__init__()
        self._dims = dims
        self._channels = channels
        self._patch_size = patch_size
        self._net = torch.nn.ConvTranspose2d(
            self._dims, self._channels, self._patch_size, self._patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._net(x)
        return x
