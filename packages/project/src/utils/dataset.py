import typing as t
import numpy as np
import pandas as pd
import dataclasses
import torch


@dataclasses.dataclass
class AugmentingTransform:
    samples: int
    transform: t.Callable[[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]


@dataclasses.dataclass
class AugmentationCache:
    spectrogram: torch.Tensor
    spectrogram_index: int
    augmenting_transform: AugmentingTransform
    augmenting_transform_index: int
    augmenting_transform_results: list[tuple[torch.Tensor, torch.Tensor]]
    augmenting_transform_results_index: int


class SpectrogramDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    __input_path: str
    __data: pd.DataFrame
    __base_transforms: t.Callable[[torch.Tensor], torch.Tensor]
    __augmenting_transforms: list[AugmentingTransform]
    __augmenting_transforms_count: int
    __cache: AugmentationCache | None

    def __init__(
        self,
        input_path: str = "./data/spectrogram",
        base_transforms: t.Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        augmenting_transforms: list[AugmentingTransform] = [],
    ):
        super().__init__()
        self.__input_path = input_path
        self.__data = pd.read_csv(f"{input_path}/dataset.csv")
        self.__base_transforms = base_transforms
        self.__augmenting_transforms = augmenting_transforms
        self.__augmenting_transforms_count = sum(
            map(lambda x: x.samples, self.__augmenting_transforms)
        )
        self.__cache = None

    def __len__(self):
        return len(self.__data) * (
            self.__augmenting_transforms_count
            if self.__augmenting_transforms_count > 0
            else 1
        )

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.__augmenting_transforms_count == 0:
            return self.__get_without_augmentations(index)

        return self.__get_with_augmentations(index)

    def __get_spectrogram(self, index: int):
        filename = self.__data.iloc[index]["spectrogram"]
        return torch.from_numpy(np.load(f"{self.__input_path}/{filename}"))

    def __get_without_augmentations(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectrogram_tensor = self.__get_spectrogram(index)
        base_transform_result = self.__base_transforms(spectrogram_tensor)

        return (spectrogram_tensor, base_transform_result, torch.tensor(0))

    def __get_with_augmentations(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_index = index // self.__augmenting_transforms_count
        augmenting_transform_index = -1
        augmenting_transform_results_index = index % self.__augmenting_transforms_count

        for i, transform in enumerate(self.__augmenting_transforms):
            if augmenting_transform_results_index - transform.samples < 0:
                augmenting_transform_index = i
                break

            augmenting_transform_results_index -= transform.samples

        if (
            self.__cache is not None
            and self.__cache.spectrogram_index == row_index
            and self.__cache.augmenting_transform_index == augmenting_transform_index
        ):
            augmenting_transform_result = self.__cache.augmenting_transform_results[
                augmenting_transform_results_index
            ]

            return (
                self.__cache.spectrogram,
                augmenting_transform_result[0],
                augmenting_transform_result[1],
            )

        spectrogram_tensor = self.__get_spectrogram(row_index)
        base_transform_result = self.__base_transforms(spectrogram_tensor)
        augmenting_transform_results = self.__augmenting_transforms[
            augmenting_transform_index
        ].transform(base_transform_result)
        augmenting_transform_result = augmenting_transform_results[
            augmenting_transform_results_index
        ]

        self.__cache = AugmentationCache(
            spectrogram=spectrogram_tensor,
            spectrogram_index=row_index,
            augmenting_transform=self.__augmenting_transforms[
                augmenting_transform_index
            ],
            augmenting_transform_index=augmenting_transform_index,
            augmenting_transform_results=augmenting_transform_results,
            augmenting_transform_results_index=augmenting_transform_results_index,
        )

        return (
            spectrogram_tensor,
            augmenting_transform_result[0],
            augmenting_transform_result[1],
        )
