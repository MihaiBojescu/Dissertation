import pandas as pd
import typing as t
import numpy as np
import torch


class SpectrogramDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    __input_path: str
    __data: pd.DataFrame
    __current_batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    __pre_transformations: t.Callable[[torch.Tensor], torch.Tensor]
    __transformations: t.Union[
        t.Callable[[torch.Tensor], torch.Tensor],
        t.Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        t.Callable[[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]],
    ]
    __multiplexing_factor: int

    def __init__(
        self,
        input_path: str = "./data/spectrogram",
        pre_transformations: t.Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        transformations: t.Union[
            t.Callable[[torch.Tensor], torch.Tensor],
            t.Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
            t.Callable[[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]],
        ] = lambda x: x,
        multiplexing_factor: int = 1,
    ):
        super().__init__()
        self.__input_path = input_path
        self.__data = pd.read_csv(f"{input_path}/dataset.csv")
        self.__current_batch = []
        self.__pre_transformations = pre_transformations
        self.__transformations = transformations
        self.__multiplexing_factor = multiplexing_factor

    def __len__(self):
        return len(self.__data) * self.__multiplexing_factor

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_index = 0
        row_index = 0

        if len(self.__current_batch) > 0:
            batch_index = index % self.__multiplexing_factor
            row_index = index // self.__multiplexing_factor

        row = self.__data.iloc[row_index]
        spectrogram = row["spectrogram"]

        if batch_index == 0 or len(self.__current_batch) == 0:
            spectrogram_tensor = torch.from_numpy(np.load(f"{self.__input_path}/{spectrogram}"))
            spectrogram_tensor = self.__pre_transformations(spectrogram_tensor)
            result = self.__transformations(spectrogram_tensor)

            if torch.is_tensor(result):
                self.__current_batch = [(result, torch.tensor(0), spectrogram_tensor)]
            elif type(result) == tuple:
                self.__current_batch = [(result[0], result[1], spectrogram_tensor)]
            elif type(result) == list:
                self.__current_batch = [
                    (entry[0], entry[1], spectrogram_tensor) for entry in result
                ]

        entry = self.__current_batch[batch_index] or (
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0),
        )

        return entry
