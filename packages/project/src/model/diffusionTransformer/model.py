import torch
from model.diffusionTransformer.patchDeEmbedding import PatchDeEmbedding
from model.diffusionTransformer.patchEmbedding import PatchEmbedding
from model.diffusionTransformer.timeEmbedding import TimeEmbedding
from model.diffusionTransformer.transformer import Transformer


class DiffusionTransformerModel(torch.nn.Module):
    _image_size: tuple[int, int]
    _patch_size: tuple[int, int]
    _tile_size: tuple[int, int]
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
        image_size: tuple[int, int] = (64, 64),
        patch_size: tuple[int, int] = (8, 8),
        n_channels: int = 3,
        embedding_dims: int = 256,
        depth: int = 4,
        n_heads: int = 3,
    ):
        super().__init__()
        self._image_size = image_size
        self._patch_size = patch_size
        self._tile_size = (
            self._image_size[0] // self._patch_size[0],
            self._image_size[1] // self._patch_size[1],
        )
        self._n_channels = n_channels
        self._embedding_dims = embedding_dims
        self._depth = depth
        self._n_heads = n_heads

        assert (
            self._image_size[0] % self._patch_size[0] == 0
            and self._image_size[1] % self._patch_size[1] == 0
        ), f"Image size ({self._image_size[0]}, {self._image_size[1]}) must divide exactly to patch size ({self._patch_size[0]}, {self._patch_size[1]}). image_size[0] ({self._image_size[0]}) % patch_size[0] ({self._patch_size[0]}) = {self._image_size[0] % self._patch_size[0]}, image_size[1] ({self._image_size[1]}) % patch_size ({self._patch_size[1]}) = {self._image_size[1] % self._patch_size[1]}"

        self._n_patches = (self._image_size[0] // self._patch_size[0]) * (
            self._image_size[1] // self._patch_size[1]
        )
        self._pos_embedding = torch.nn.Parameter(
            torch.zeros(1, self._n_patches, self._embedding_dims)
        )
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

    def forward(self, x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        batch_size, color_channels, height, width = x.shape
        x_embedding = self._patch_embedding(x)
        x_embedding = x_embedding.flatten(2).transpose(1, 2)
        x_embedding = x_embedding + self._pos_embedding

        t_embedding = self._time_embedding(step)
        x_embedding = x_embedding + t_embedding[:, None, :]

        x = self._transformers(x_embedding)

        x = x.transpose(1, 2).unflatten(2, self._tile_size)
        x = self._patch_de_embedding(x)
        x = x.view(batch_size, color_channels, height, width)

        return x
