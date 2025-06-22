import os
import typing as t
import scipy.io.wavfile as wav
import torch
from interfaces.dataset.lastEntry import LastEntry


class WavDataset(torch.utils.data.Dataset[torch.Tensor]):
    __input_path: str
    __transformations: t.Callable[[t.Any], torch.Tensor]
    __files: list[str]
    __last_entry: LastEntry[torch.Tensor] | None

    def __init__(
        self,
        input_path: str = "./data/input",
        transformations: t.Callable[[t.Any], torch.Tensor] = lambda x: x,
    ):
        super().__init__()
        self.__input_path = input_path
        self.__transformations = transformations
        self.__files = os.listdir(self.__input_path)
        self.__files = [entry for entry in self.__files if os.path.splitext(entry)[1] == ".wav"]
        self.__last_entry = None

    def __len__(self):
        return len(self.__files)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.__last_entry and index == self.__last_entry.index:
            return self.__last_entry.data

        sample_rate, data = wav.read(f"{self.__input_path}/{self.__files[index]}")
        result = self.__transformations((sample_rate, data))

        self.__last_entry = LastEntry(index=index, data=result)
        return result
