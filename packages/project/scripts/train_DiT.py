#!/usr/bin/env python3

import datetime
import sys
import os
import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
torch.multiprocessing.set_start_method("spawn", force=True)

import torchvision.transforms.v2 as v2
from utils.endNoisify import EndNoisify
from utils.device import get_device
from utils.randomRangeNoisify import RandomRangeNoisify
from utils.dataset import SpectrogramDataset, AugmentingTransform
from utils.collate import PaddingCollate
from training.trainer import ModelTrainer
from model.diffusionTransformer.model import DiffusionTransformerModel


def main():
    device = get_device()
    torch.set_default_device(device)

    base_transformations = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((412, 256))]
    )
    augmenting_transformations = [
        AugmentingTransform(samples=10, transform=RandomRangeNoisify(10)),
        AugmentingTransform(
            samples=10, transform=EndNoisify(samples=10, max_percentage=0.5)
        ),
    ]

    dataset = SpectrogramDataset(
        base_transforms=base_transformations,
        augmenting_transforms=augmenting_transformations,
    )
    collate = PaddingCollate()
    generator = torch.Generator(device)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate.collate,
        generator=generator,
    )

    model = DiffusionTransformerModel(
        image_size=(412, 256),
        patch_size=(4, 4),
        n_channels=2,
        embedding_dims=256,
        depth=4,
        n_heads=4,
    )
    optimiser = torch.optim.Adam(model.parameters(), 0.005)
    loss_function = torch.nn.MSELoss()

    trainer = ModelTrainer(model, optimiser, loss_function)
    trainer.train(dataloader, epochs=5)
    trainer.save(
        f"./data/weights/{datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()}.pt"
    )


if __name__ == "__main__":
    main()
