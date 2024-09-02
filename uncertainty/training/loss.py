import torch
from torch import nn
from torch.nn.functional import sigmoid

class SmoothDiceLoss(nn.Module):
    def __init__(self):
        super(SmoothDiceLoss, self).__init__()
    
    def dice(self, y_pred: torch.Tensor, y: torch.Tensor, logits: bool, smooth: int):
        if logits:
            y_pred = sigmoid(y_pred)
        intersection = (y_pred.flatten() * y.flatten()).sum()
        return (2 * intersection + smooth) / (y_pred.sum() + y.sum() + smooth)

    def generalised_dice(self, y_preds: torch.Tensor, ys: torch.Tensor, logits: bool, smooth: int):
        """
        Input (C, ...), targets (C, ...)

        average for now
        """
        return torch.mean(
            torch.stack([self.dice(y_preds[i], ys[i], logits, smooth) for i in range(y_preds.shape[0])]), dim=0
        )

    def forward(self, y_preds: torch.Tensor, ys, logits: bool = True, smooth: int=1):
        """
        Input (N, C, ...), targets (C, ...)

        must pass through final activation

        If True, we apply sigmoid
        """
        return 1 - torch.mean(
            torch.stack([self.generalised_dice(y_preds[i], ys[i], logits, smooth) for i in range(y_preds.shape[0])]), dim=0
        )
