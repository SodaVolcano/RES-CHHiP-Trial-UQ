"""
Wrapper around a model that produces multiple outputs using test-time augmentation (TTA)
"""

from torch import nn
import torch
import torchio as tio
from ..training.augmentations import inverse_affine_transform
from kornia.augmentation import RandomAffine3D
import toolz as tz
from toolz import curried


class TTA(nn.Module):
    """
    Produce multiple outputs using test-time augmentation (TTA)
    """

    def __init__(
        self, model: nn.Module, aug: tio.Compose, batch_affine: RandomAffine3D
    ):
        super().__init__()
        self.model = model
        self.aug = aug
        self.batch_affine = batch_affine

    @torch.no_grad()
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single input of shape (B, C, D, H, W)

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, C', D, H, W) where C' is the number of classes
            in the output
        """
        xs_subject = [tio.Subject(volume=tio.ScalarImage(tensor=x_i)) for x_i in x]
        y_preds = tz.pipe(
            xs_subject,
            curried.map(self.aug),
            curried.map(lambda subj: subj["volume"].data),
            list,
            torch.stack,
            lambda x: self.batch_affine(x, return_transform=True),
            self.model,
            inverse_affine_transform(self.batch_affine._params),
        )
        for y_pred, subj in zip(y_preds, xs_subject):
            subj["mask"] = tio.LabelMap(tensor=y_pred)
        return torch.stack(
            [
                subj.apply_inverse_transform(warn=False)["mask"].data
                for subj in xs_subject
            ]
        )

    def forward(self, x: torch.Tensor, n_forwards: int = 10) -> torch.Tensor:
        """
        Forward pass for a batch of inputs of shape (B, C, D, H, W)

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, n_outputs, C', D, H, W) where C' is the number of classes
            in the output
        """
        return torch.stack([self.forward_single(x) for _ in range(n_forwards)], dim=1)
