import torch


class DenoiseCNN(torch.nn.Module):
    _encode_1: torch.nn.Module
    _pool_1: torch.nn.Module
    _encode_2: torch.nn.Module
    _pool_2: torch.nn.Module
    _bottleneck: torch.nn.Module
    _time_embedding: torch.nn.Module
    _upscale_2: torch.nn.Module
    _decode_2: torch.nn.Module
    _upscale_1: torch.nn.Module
    _decode_1: torch.nn.Module
    _out_conv: torch.nn.Module

    def __init__(self, in_channels: int = 2, base_feats: int = 64):
        super().__init__()

        self._encode_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, base_feats, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_feats, base_feats, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self._pool_1 = torch.nn.MaxPool2d(2)

        self._encode_2 = torch.nn.Sequential(
            torch.nn.Conv2d(base_feats, base_feats * 2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_feats * 2, base_feats * 2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self._pool_2 = torch.nn.MaxPool2d(2)

        self._bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(base_feats * 2, base_feats * 4, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_feats * 4, base_feats * 4, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self._time_embedding = torch.nn.Sequential(
            torch.nn.Linear(1, base_feats * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(base_feats * 4, base_feats * 4),
        )

        self._upscale_2 = torch.nn.ConvTranspose2d(
            base_feats * 4, base_feats * 2, kernel_size=2, stride=2
        )
        self._decode_2 = torch.nn.Sequential(
            torch.nn.Conv2d(base_feats * 4, base_feats * 2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_feats * 2, base_feats * 2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        self._upscale_1 = torch.nn.ConvTranspose2d(
            base_feats * 2, base_feats, kernel_size=2, stride=2
        )
        self._decode_1 = torch.nn.Sequential(
            torch.nn.Conv2d(base_feats * 2, base_feats, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_feats, base_feats, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        self._out_conv = torch.nn.Conv2d(base_feats, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        encoded_1 = self._encode_1(x)
        pooled_1 = self._pool_1(encoded_1)

        encoded_2 = self._encode_2(pooled_1)
        pooled_2 = self._pool_2(encoded_2)

        bottlenecked = self._bottleneck(pooled_2)

        if step.ndim == 0:
            step = step.unsqueeze(0)
        step = step.view(batch_size, 1).float()
        t_embedding = self._time_embedding(step)
        t_embedding = t_embedding.view(batch_size, -1, 1, 1)

        bottlenecked = bottlenecked + t_embedding

        upscaled_2 = self._upscale_2(bottlenecked)
        concatenated_2 = torch.cat([upscaled_2, encoded_2], dim=1)
        decoded_2 = self._decode_2(concatenated_2)

        upscaled_1 = self._upscale_1(decoded_2)
        concatenated_1 = torch.cat([upscaled_1, encoded_1], dim=1)
        decoded_1 = self._decode_1(concatenated_1)

        out = self._out_conv(decoded_1)
        return out
