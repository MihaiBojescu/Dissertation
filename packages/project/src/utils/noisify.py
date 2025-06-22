import torch

from interfaces.utils.transform import MultiDiffusionTransform


class BaseNoisify(MultiDiffusionTransform[list[tuple[torch.Tensor, torch.Tensor]]]):
    _beta_start: float
    _beta_end: float
    _alphas_cumprod: torch.Tensor

    def __init__(
        self, samples: int, beta_start: float = 0.0025, beta_end: float = 0.05
    ):
        super().__init__(samples)
        self._beta_start = beta_start
        self._beta_end = beta_end

        betas = torch.linspace(beta_start, beta_end, samples)
        alphas = 1 - betas

        self._alphas_cumprod = torch.cumprod(alphas, dim=0)

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return []

    def _noisify(
        self, x: torch.Tensor, start: int, end: int, step: int
    ) -> torch.Tensor:
        x = x.clone()

        sqrt_alpha = torch.sqrt(self._alphas_cumprod[step])
        sqrt_one_minus_alpha = torch.sqrt(1 - self._alphas_cumprod[step])
        x_noise = torch.randn((x.shape[-2], end - start))

        result = x[..., start:end] * sqrt_alpha + x_noise * sqrt_one_minus_alpha
        result = torch.clip(result, 0, 1)

        x[..., start:end] = result

        return x
