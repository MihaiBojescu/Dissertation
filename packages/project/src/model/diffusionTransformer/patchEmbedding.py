import torch


class PatchEmbedding(torch.nn.Module):
    _dims: int
    _channels: int
    _patch_size: tuple[int, int]
    _device: torch.device
    _net: torch.nn.Module

    def __init__(
        self,
        dims: int,
        channels: int,
        patch_size: tuple[int, int],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self._dims = dims
        self._channels = channels
        self._patch_size = patch_size
        self._device = device

        self._net = torch.nn.Conv2d(self._channels, self._dims, patch_size, patch_size)
        self._net = self._net.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._net(x)
        return x
