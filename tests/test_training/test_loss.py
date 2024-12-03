import torch
from torch import nn

from ..context import training

ConfidNetMSELoss = training.ConfidNetMSELoss
DeepSupervisionLoss = training.DeepSupervisionLoss
DiceBCELoss = training.DiceBCELoss
SmoothDiceLoss = training.SmoothDiceLoss


class TestConfidNetMSELoss:

    # Compute MSE loss between predicted confidence and actual class probability
    def test_mse_loss_computation(self):
        # Initialize loss function
        loss_fn = ConfidNetMSELoss(weighting=1.5)

        # Create sample inputs
        confidence = torch.tensor(
            [[0.8, 0.2, 0.3, 0.6], [0.7, 0.1, 0.9, 0.1]], requires_grad=True
        )
        # confidence version that predicts 0 for `0` classes
        confidence2 = torch.tensor(
            [[0.8, 0.0, 0.0, 0.6], [0.7, 0.5, 0.9, 0.0]], requires_grad=True
        )
        y_pred = torch.tensor(
            [[0.8, 0.2, 0.3, 0.6], [0.7, 0.5, 0.9, 0.4]], requires_grad=True
        )
        y = torch.tensor([[1, 0, 0, 1], [1, 1, 1, 0]])

        # Compute loss
        loss1 = loss_fn((confidence, y_pred), y)
        loss2 = loss_fn((confidence2, y_pred), y)

        assert torch.allclose(torch.tensor(0.1900), loss1, rtol=1e-4)
        assert torch.allclose(torch.tensor(0.0), loss2, rtol=1e-4)
        assert loss1.requires_grad and loss2.requires_grad


class TestDeepSupervisionLoss:

    # Loss calculation with multiple U-Net levels using default weights
    def test_multi_level_loss_calculation(self):
        # Create mock loss function
        mock_loss = nn.MSELoss()

        # Initialize deep supervision loss
        deep_loss = DeepSupervisionLoss(mock_loss)

        # Create sample predictions at different resolutions
        torch.manual_seed(0)
        y_pred1 = torch.randn(2, 1, 16, 16)
        y_pred2 = torch.randn(2, 1, 32, 32)
        y_pred3 = torch.randn(2, 1, 64, 64)
        y_preds = [y_pred1, y_pred2, y_pred3]

        # Create ground truth
        y = torch.randn(2, 1, 64, 64, requires_grad=True)

        # Calculate loss
        loss = deep_loss(y_preds, y)

        assert torch.allclose(loss, torch.tensor(2.0205), atol=1e-4)
        assert loss.requires_grad


class TestDiceBCELoss:

    # Combined BCE and Dice loss calculation with logits input
    def test_combined_loss_with_logits(self):
        # Create test tensors
        y_pred = torch.tensor([[0.8, 0.4], [0.3, 0.9]], requires_grad=True)
        y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        # Initialize loss function
        loss_fn = DiceBCELoss()

        # Calculate combined loss
        combined_loss = loss_fn(y_pred, y, separate=False, logits=True)

        assert torch.allclose(combined_loss, torch.tensor(0.8602), atol=1e-2)
        assert combined_loss.requires_grad


class TestSmoothDiceLoss:

    # Compute Dice loss for single channel binary segmentation with logits input
    def test_single_channel_binary_segmentation(self):
        # Initialize loss function
        dice_loss = SmoothDiceLoss(smooth=1)

        # Create sample prediction and target tensors
        y_pred = torch.tensor(
            [[[[0.8, 0.2], [0.3, 0.9]]]], requires_grad=True
        )  # Shape: (1,1,2,2)
        y_true = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # Shape: (1,1,2,2)

        # Compute loss
        loss = dice_loss(y_pred, y_true, logits=True)

        assert torch.allclose(loss, torch.tensor(0.3119), atol=1e-4)
        assert loss.requires_grad
