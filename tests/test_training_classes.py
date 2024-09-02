import torch
from torch import nn
from .context import uncertainty

H5Dataset = uncertainty.training.datasets.H5Dataset
LitSegmentation = uncertainty.training.LitSegmentation
PREPROCESS_DATA_CONFIGURABLE_PATCH = (
    "uncertainty.training.classes._preprocess_data_configurable"
)


class TestLitSegmentation:
    # Model initialization with default parameters
    def test_initialization_with_default_parameters(self, mocker):
        model = mocker.Mock(spec=nn.Module)
        config = {
            "output_channel": 3,
            "optimiser": mocker.Mock(),
            "optimiser_kwargs": {},
            "lr_scheduler": mocker.Mock(),
        }
        lit_segmentation = LitSegmentation(model=model, config=config)  # type: ignore
        assert lit_segmentation.model == model
        assert lit_segmentation.config == config
        assert lit_segmentation.deep_supervision is False
        assert lit_segmentation.class_weights is None

    # Calculation of loss without deep supervision
    def test_calculation_loss_without_deep_supervision(self, mocker):
        model = mocker.Mock(spec=nn.Module)
        model.last_activation = nn.Sigmoid()
        config = {
            "output_channel": 3,
            "optimiser": mocker.Mock(),
            "optimiser_kwargs": {},
            "lr_scheduler": mocker.Mock(),
        }
        lit_segmentation = LitSegmentation(model=model, config=config)  # type: ignore

        y_pred = torch.randn(
            3, config["output_channel"], 100, 100, 100
        )  # Example random prediction tensor
        y = torch.randint(
            0, 2, (3, config["output_channel"], 100, 100, 100), dtype=torch.float32
        )  # Example random ground truth tensor

        loss = lit_segmentation.calc_loss(y_pred, y)

        assert isinstance(loss, torch.Tensor)

    # Calculation of loss with deep supervision
    def test_calculation_loss_deep_supervision(self, mocker):
        model = mocker.Mock(spec=nn.Module)
        model.last_activation = nn.Sigmoid()
        model.deep_supervision = True
        config = {
            "output_channel": 3,
            "optimiser": mocker.Mock(),
            "optimiser_kwargs": {},
            "lr_scheduler": mocker.Mock(),
        }
        lit_segmentation = LitSegmentation(model=model, config=config)  # type: ignore
        y_preds = [
            torch.randn(3, config["output_channel"], 32, 32, 32),
            torch.randn(3, config["output_channel"], 64, 64, 64),
            torch.randn(3, config["output_channel"], 128, 128, 128),
        ]
        y = torch.randint(
            0, 2, (3, config["output_channel"], 128, 128, 128), dtype=torch.float32
        )
        loss = lit_segmentation._LitSegmentation__calc_loss_deep_supervision(y_preds, y)
        assert isinstance(loss, torch.Tensor)
