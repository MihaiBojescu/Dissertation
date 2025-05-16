import torch


class PaddingCollate:
    def collate(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, steps, labels = zip(*batch)
        max_h = max(img.shape[-2] for img in images)
        max_w = max(img.shape[-1] for img in images)

        padded_images = [
            torch.nn.functional.pad(
                image, (0, max_w - image.shape[-1], 0, max_h - image.shape[-2])
            )
            for image in images
        ]
        padded_labels = [
            torch.nn.functional.pad(
                label, (0, max_w - label.shape[-1], 0, max_h - label.shape[-2])
            )
            for label in labels
        ]

        return (
            torch.stack(padded_images, dim=0),
            torch.stack(steps, dim=0),
            torch.stack(padded_labels, dim=0),
        )
