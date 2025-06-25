import torch


class PaddingCollate:
    def collate(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, step = zip(*batch)
        max_h = max(img.shape[-2] for img in x)
        max_w = max(img.shape[-1] for img in x)

        padded_x = [
            torch.nn.functional.pad(
                image, (0, max_w - image.shape[-1], 0, max_h - image.shape[-2])
            )
            for image in x
        ]
        padded_y = [
            torch.nn.functional.pad(
                label, (0, max_w - label.shape[-1], 0, max_h - label.shape[-2])
            )
            for label in y
        ]

        return (
            torch.stack(padded_x, dim=0),
            torch.stack(padded_y, dim=0),
            torch.stack(step, dim=0),
        )
