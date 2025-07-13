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
from torch.utils.tensorboard import SummaryWriter
from utils.endNoisify import EndNoisify
from utils.device import get_device
from utils.randomRangeNoisify import RandomRangeNoisify
from utils.dataset import SpectrogramDataset, AugmentingTransform
from utils.collate import PaddingCollate
from training.trainer import ModelTrainer
from model.denoiseCnn.model import DenoiseCNN


def main():
    device = get_device()

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
        cache_size=16,
    )
    collate = PaddingCollate()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate.collate,
        num_workers=8,
        pin_memory=True,
    )

    model = DenoiseCNN(in_channels=2, base_feats=32, device=device)
    optimiser = torch.optim.Adam(model.parameters(), 0.005)
    loss_function = torch.nn.MSELoss()
    summary_writer = SummaryWriter("./logs")

    trainer = ModelTrainer(model, optimiser, loss_function, device)
    trainer.train(
        dataloader,
        epochs=5,
        callback=lambda epoch, entry, _y_hat, loss: summary_writer.add_scalar(
            "Loss/train", loss, epoch * len(dataloader) + entry
        ),
    )
    trainer.eval(
        dataloader,
        callback=lambda epoch, entry, _y_hat, loss: summary_writer.add_scalar(
            "Loss/eval", loss, epoch * len(dataloader) + entry
        ),
    )
    summary_writer.flush()
    trainer.save(
        f"./data/weights/CNN_{datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()}.pt"
    )


if __name__ == "__main__":
    main()
