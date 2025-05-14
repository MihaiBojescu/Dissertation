import pandas as pd
import torch
import typing as t


class SpectrogramDataset(torch.utils.data.Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(self, path: str, transformations: t.Callable[[t.Any], torch.Tensor]):
        super().__init__()
        self.__data = pd.read_csv(path)
        self.__transformations = transformations

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.data.iloc[index]
        image = row["image"]
        channel = int(row["channel"])
        step = int(row["step"])

        transformed_image = self.__transformations(image)

        return (transformed_image, channel, step)
