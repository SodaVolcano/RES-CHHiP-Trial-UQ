import torch
from torch import nn
from torch.nn.functional import sigmoid


class SmoothDiceLoss(nn.Module):
    def __init__(self, smooth: int = 1):
        super(SmoothDiceLoss, self).__init__()
        self.smooth = smooth

    def dice(self, y_pred: torch.Tensor, y: torch.Tensor, logits: bool):
        if logits:
            y_pred = sigmoid(y_pred)
        intersection = (y_pred.flatten() * y.flatten()).sum()
        return (2 * intersection + self.smooth) / (y_pred.sum() + y.sum() + self.smooth)

    def generalised_dice(self, y_preds: torch.Tensor, ys: torch.Tensor, logits: bool):
        """
        Input (C, ...), targets (C, ...)

        average for now
        """
        return torch.mean(
            torch.stack(
                [self.dice(y_preds[i], ys[i], logits) for i in range(y_preds.shape[0])]
            ),
            dim=0,
        )

    def forward(self, y_preds: torch.Tensor, ys, logits: bool = True):
        """
        Input (N, C, ...), targets (C, ...)

        must pass through final activation

        If True, we apply sigmoid
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
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        logits: bool = True,
        smooth: int = 1,
    ):
        super().__init__()
        self.bce = (
            nn.BCEWithLogitsLoss(class_weights)
            if logits is not None
            else nn.BCELoss(class_weights)
        )
        self.dice = SmoothDiceLoss(smooth)

    def forward(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        separate: bool = False,
    ):
        """
        Parameters
        ----------
        separate : bool
            If True, return both BCE and Dice losses separately in that order
        """
        bce = self.bce(y_pred, y)
        dice = self.dice(y_pred, y, True)
        return (bce, dice) if separate else bce + dice


class DeepSupervisionLoss(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, y_preds: list[torch.Tensor], y: torch.Tensor) -> int:
        """
        Calculate the loss for each level of the U-Net

        Parameters
        ----------
        y_preds : list[torch.Tensor]
            List of outputs from the U-Net ordered from the lowest resolution
            to the highest resolution
        """
        ys = [nn.Upsample(size=y_pred.shape[2:])(y) for y_pred in y_preds]

        weights = [1 / (2**i) for i in range(len(y_preds))]
        weights = [weight / sum(weights) for weight in weights]  # normalise to sum to 1

        # Reverse weight to match y_preds; halves weights as resolution decreases
        return sum(
            weight * self.loss(y_pred, y_scaled)
            for weight, y_pred, y_scaled in zip(reversed(weights), y_preds, ys)
        )
