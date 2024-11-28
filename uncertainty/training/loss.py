"""
Training loss functions for segmentation models.
"""

import torch
from torch import nn
from torch.nn.functional import sigmoid


class SmoothDiceLoss(nn.Module):
    """
    Differentiable Dice loss, defined as 1 - Dice

    Parameters
    ----------
    smooth : int
        Smoothing factor for Dice loss to prevent division by zero
    """

    def __init__(self, smooth: int = 1):
        super(SmoothDiceLoss, self).__init__()
        self.smooth = smooth

    def dice(self, y_pred: torch.Tensor, y: torch.Tensor, logits: bool):
        """
        Compute dice for a single channel between y_pred and y of shape (H, W, D)
        """
        if logits:
            y_pred = sigmoid(y_pred)
        intersection = (y_pred.flatten() * y.flatten()).sum()
        return (2 * intersection + self.smooth) / (y_pred.sum() + y.sum() + self.smooth)

    def generalised_dice(self, y_pred: torch.Tensor, y: torch.Tensor, logits: bool):
        """
        Compute average dice for all channels between y_pred and y of shape (C, H, W, D)
        """
        return torch.mean(
            torch.stack(
                [self.dice(y_pred[i], y[i], logits) for i in range(y_pred.shape[0])]
            ),
            dim=0,
        )

    def forward(self, y_preds: torch.Tensor, ys, logits: bool = True):
        """
        Compute average smooth dice loss on batch of predictions and targets

        Parameters
        ----------
        y_preds : torch.Tensor
            Predictions of shape (N, C, H, W, D)
        ys : torch.Tensor
            Targets of shape (N, C, H, W, D)
        logits : bool
            Whether predictions are logits or probabilities, if logits, sigmoid is applied
        """
        return 1 - torch.mean(
            torch.stack(
                [
                    self.generalised_dice(y_preds[i], ys[i], logits)
                    for i in range(y_preds.shape[0])
                ]
            ),
            dim=0,
        )


class DiceBCELoss(nn.Module):
    """
    Dice + BCE loss, defined as BCE + DiceLoss

    Parameters
    ----------
    class_weights : torch.Tensor
        Optional weights to apply to each class
    smooth : int
        Smoothing factor for Dice loss to prevent division by zero
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        smooth: int = 1,
    ):
        super().__init__()
        self.bce = nn.BCELoss(class_weights)
        self.bce_logits = nn.BCEWithLogitsLoss(class_weights)
        self.dice = SmoothDiceLoss(smooth)

    def forward(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        separate: bool = False,
        logits: bool = True,
    ):
        """
        Parameters
        ----------
        separate : bool
            If True, return both BCE and Dice losses separately in that order
        logits : bool
            Whether predictions are logits or probabilities, if logits, sigmoid is applied
        """
        bce = self.bce_logits(y_pred, y) if logits else self.bce(y_pred, y)
        dice = self.dice(y_pred, y, logits)
        return (bce, dice) if separate else bce + dice


class DeepSupervisionLoss(nn.Module):
    """
    Deep supervision loss, defined as the sum of losses at each level of the U-Net

    Parameters
    ----------
    loss : nn.Module
        Loss function that's applied to each level of the U-Net. Deep supervision
        computes a weighted sum of these losses for each level.
    """

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(
        self, y_preds: list[torch.Tensor], y: torch.Tensor, **loss_kwargs
    ) -> int:
        """
        Calculate the loss for each level of the U-Net

        Parameters
        ----------
        y_preds : list[torch.Tensor]
            List of outputs from the U-Net ordered from the lowest resolution
            to the highest resolution
        y : torch.Tensor
            Ground truth target
        loss_kwargs : dict
            Additional keyword arguments to pass to the loss
        """
        ys = [nn.Upsample(size=y_pred.shape[2:])(y) for y_pred in y_preds]

        weights = [1 / (2**i) for i in range(len(y_preds))]
        weights = [weight / sum(weights) for weight in weights]  # normalise to sum to 1

        # Reverse weight to match y_preds; halves weights as resolution decreases
        return sum(
            weight * self.loss(y_pred, y_scaled, **loss_kwargs)
            for weight, y_pred, y_scaled in zip(reversed(weights), y_preds, ys)
        )


class ConfidNetMSELoss(nn.Module):
    """
    Mean squared error loss for confidence prediction in ConfidNet

    Taken from https://github.com/valeoai/ConfidNet

    Parameters
    ----------
    weighting : float
        Weighting factor for misclassified samples
    """

    def __init__(self, weighting: float = 1.5):
        super().__init__()
        self.weighting = weighting

    def forward(self, conf_pred: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor):
        """
        Compute MSE loss between predicted and actual class probability/confidence

        Parameters
        ----------
        conf_pred : tuple[torch.Tensor, torch.Tensor]
            Predicted (confidence, class probability) from ConfidNet
        y : torch.Tensor
            Ground truth target
        """
        confidence, y_pred = conf_pred
        weights = torch.ones_like(y, dtype=torch.float)
        weights[((y_pred > 0.5).to(torch.float) != y)] *= self.weighting
        return torch.mean((weights * (confidence - (y_pred * y)) ** 2).sum(dim=1))
