import pandas as pd
import typing as t
import torch
import torchvision.io as io


class SpectrogramDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    __input_path: str
    __data: pd.DataFrame
    __transformations: t.Union[
        t.Callable[[torch.Tensor], torch.Tensor],
        t.Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        t.Callable[[torch.Tensor], tuple[list[torch.Tensor], torch.Tensor]],
    ]

    def __init__(
        self,
        input_path: str = "./data/spectrogram",
        transformations: t.Union[
            t.Callable[[torch.Tensor], torch.Tensor],
            t.Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
            t.Callable[[torch.Tensor], tuple[list[torch.Tensor], torch.Tensor]],
        ] = lambda x: x,
    ):
        super().__init__()
        self.__input_path = input_path
        self.__data = pd.read_csv(f"{input_path}/dataset.csv")
        self.__transformations = transformations

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.__data.iloc[index]
        filename = row["filename"]
        channel = torch.tensor(row["channel"])

        image_tensor = io.decode_image(
            f"{self.__input_path}/{filename}", mode=io.ImageReadMode.RGB
        )
        image_tensor = self.__transformations(image_tensor)

        return image_tensor, channel
