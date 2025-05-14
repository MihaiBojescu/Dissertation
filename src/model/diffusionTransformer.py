import torch
from model.patchDeEmbedding import PatchDeEmbedding
from model.patchEmbedding import PatchEmbedding
from model.timeEmbedding import TimeEmbedding
from model.transformer import Transformer


class DiffusionTransformer(torch.nn.Module):
    _image_size: int
    _patch_size: int
    _n_channels: int
    _embedding_dims: int
    _depth: int
    _n_heads: int
    _n_patches: int
    _patch_embedding: torch.nn.Module
    _time_embedding: torch.nn.Module
    _transformers: torch.nn.Module
    _patch_de_embedding: torch.nn.Module

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 64,
        n_channels: int = 3,
        embedding_dims: int = 256,
        depth: int = 4,
        n_heads: int = 3,
    ):
        super().__init__()
        self._image_size = image_size
        self._patch_size = patch_size
        self._n_channels = n_channels
        self._embedding_dims = embedding_dims
        self._depth = depth
        self._n_heads = n_heads

        assert (
            self._image_size % self._patch_size == 0
        ), f"Image size ({self._image_size}) must divide exactly to patch size ({self._patch_size}). Image size ({self._image_size}) % patch size ({self._patch_size}) = {self._image_size % self._patch_size} !== 0"

        self._n_patches = (self._image_size // self._patch_size) ** 2
        self._patch_embedding = PatchEmbedding(
            self._embedding_dims, self._n_channels, self._patch_size
        )
        self._time_embedding = TimeEmbedding(self._embedding_dims)
        self._transformers = torch.nn.Sequential(
            *(
                Transformer(self._embedding_dims, self._n_heads)
                for _ in range(self._depth)
            )
        )
        self._patch_de_embedding = PatchDeEmbedding(
            self._embedding_dims, self._n_channels, self._patch_size
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size, color_channels, height, width = x.shape
        x_embedding = self._patch_embedding(x)
        x_embedding = x_embedding.flatten(2).transpose(1, 2)

        t_embedding = self._time_embedding(t)
        x_embedding = x_embedding + t_embedding[:, None, :]

        x = self._transformers(x_embedding)
        x = self._patch_de_embedding(x)

        return x.transpose(1, 2).view(batch_size, color_channels, height, width)
