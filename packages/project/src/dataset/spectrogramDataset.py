import torch
from interfaces.dataset.lastEntry import LastEntry
from interfaces.utils.transform import MultiDiffusionTransform


class SpectrogramDiffusionDataset(
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    __input_dataset: torch.utils.data.Dataset[torch.Tensor]
    __transformations: list[
        MultiDiffusionTransform[list[tuple[torch.Tensor, torch.Tensor]]]
    ]
    __transformations_count: int
    __last_entry: (
        LastEntry[tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]] | None
    )

    def __init__(
        self,
        input_dataset: torch.utils.data.Dataset[torch.Tensor],
        transformations: list[
            MultiDiffusionTransform[list[tuple[torch.Tensor, torch.Tensor]]]
        ] = [],
    ):
        super().__init__()
        self.__input_dataset = input_dataset
        self.__transformations = transformations
        self.__transformations_count = sum(
            entry.samples for entry in self.__transformations
        )
        self.__last_entry = None

    def __len__(self):
        return len(self.__input_dataset) * self.__transformations_count

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_dataset_index = index
        batch_index = 0
        transformation_index = 0
        transformation: (
            MultiDiffusionTransform[list[tuple[torch.Tensor, torch.Tensor]]] | None
        ) = None

        if self.__transformations_count != 0:
            input_dataset_index = index // self.__transformations_count

        if self.__transformations_count != 0:
            batch_index = index % self.__transformations_count

        transformation_counter = 0
        for entry in self.__transformations:
            if transformation_counter + entry.samples > batch_index:
                break

            transformation_index += 1
            transformation = entry
            transformation_counter += entry.samples

        if self.__last_entry and index == self.__last_entry.index:
            return (
                self.__last_entry.data[0][batch_index][0],
                self.__last_entry.data[0][batch_index][1],
                self.__last_entry.data[1],
            )

        entry = self.__input_dataset[input_dataset_index]
        transformed_results = [(entry, torch.tensor(0))]

        if transformation:
            transformed_results = transformation(entry)

        if batch_index == 0:
            self.__last_entry = LastEntry(
                index=index, data=(transformed_results, entry)
            )

        return (
            transformed_results[batch_index][0],
            transformed_results[batch_index][1],
            entry,
        )
