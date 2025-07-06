import unittest
import torch
from ..context import src


class TestSpectrogramDataset(unittest.TestCase):
    """
    Tests the SpectrogramDataset class. Mainly used to ensure that no out-of-bounds issues can arise.
    """

    def test_length(self):
        dataset = src.utils.dataset.SpectrogramDataset()
        self.assertGreater(len(dataset), 0)

    def test_access_first(self):
        dataset = src.utils.dataset.SpectrogramDataset()
        self.assertIsNotNone(dataset[0])

    def test_access_last(self):
        dataset = src.utils.dataset.SpectrogramDataset()
        self.assertIsNotNone(dataset[len(dataset) - 1])

    def test_access_first_with_augmentations(self):
        dataset = src.utils.dataset.SpectrogramDataset(
            augmenting_transforms=[
                src.utils.dataset.AugmentingTransform(
                    samples=10,
                    transform=lambda x: [
                        (x, torch.tensor(0)),
                        (x, torch.tensor(1)),
                        (x, torch.tensor(2)),
                        (x, torch.tensor(3)),
                        (x, torch.tensor(4)),
                        (x, torch.tensor(5)),
                        (x, torch.tensor(6)),
                        (x, torch.tensor(7)),
                        (x, torch.tensor(8)),
                        (x, torch.tensor(9)),
                    ],
                )
            ]
        )
        self.assertIsNotNone(dataset[0])

    def test_access_last_with_augmentations(self):
        dataset = src.utils.dataset.SpectrogramDataset(
            augmenting_transforms=[
                src.utils.dataset.AugmentingTransform(
                    samples=10,
                    transform=lambda x: [
                        (x, torch.tensor(0)),
                        (x, torch.tensor(1)),
                        (x, torch.tensor(2)),
                        (x, torch.tensor(3)),
                        (x, torch.tensor(4)),
                        (x, torch.tensor(5)),
                        (x, torch.tensor(6)),
                        (x, torch.tensor(7)),
                        (x, torch.tensor(8)),
                        (x, torch.tensor(9)),
                    ],
                )
            ]
        )
        self.assertIsNotNone(dataset[len(dataset) - 1])
