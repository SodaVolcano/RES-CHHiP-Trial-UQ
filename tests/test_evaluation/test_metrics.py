from uncertainty.evaluation.metrics import rc_curve_stats
from ..context import evaluation
import torch
from torch.nn.functional import sigmoid
import numpy as np
from torchmetrics.classification import MultilabelF1Score

_average_methods = evaluation.metrics._average_methods
_prepare_tensors = evaluation.metrics._prepare_tensors
dice = evaluation.dice
surface_dice = evaluation.surface_dice
hausdorff_distance = evaluation.hausdorff_distance
generalised_energy_distance = evaluation.generalised_energy_distance
rc_curve_stats = evaluation.rc_curve_stats
aurc = evaluation.aurc
eaurc = evaluation.eaurc

PRED_2D = torch.tensor(
    [
        [[0.8089, 0.2420], [0.2009, 0.0991], [0.4544, 0.2595], [0.5912, 0.1188]],
        [[0.0325, 0.4837], [0.1650, 0.8039], [0.9402, 0.3407], [0.8378, 0.3894]],
        [[0.9712, 0.5858], [0.1586, 0.0999], [0.7737, 0.8748], [0.0797, 0.8626]],
    ]
)

LABEL_2D = torch.tensor(
    [
        [[1, 1], [0, 0], [1, 1], [1, 1]],
        [[0, 0], [0, 0], [1, 1], [1, 0]],
        [[0, 1], [1, 0], [0, 0], [0, 0]],
    ]
)


def get_3d_predictions():
    np.random.seed(69)
    prediction = torch.tensor(np.random.rand(3, 4, 6, 3))
    np.random.seed(69)
    label = torch.tensor(np.random.randint(0, 2, (3, 4, 6, 3)))
    return prediction, label


class Test_AverageMethods:
    preds = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]])
    labels = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 0], [0, 1, 0, 1]])
    simple_metric = lambda self, tensor1, tensor2: MultilabelF1Score(
        tensor1.shape[0], average="micro"
    )(tensor1.unsqueeze(0), tensor2.unsqueeze(0))

    # The function returns a callable that computes metrics correctly for "micro" averaging
    def test_micro_averaging(self):
        # Get the micro averaging function
        micro_avg_func = _average_methods("micro")

        # Call the function and get the result
        result = micro_avg_func(self.preds, self.labels, self.simple_metric)

        # Assert the result is as expected
        expected_result = torch.tensor(0.5455)
        assert torch.allclose(result, expected_result, 1e-4)

    # The function returns a callable that computes metrics correctly for "macro" averaging
    def test_macro_averaging(self):

        # Expected macro average result
        expected_macro_average = torch.tensor(0.5556)

        # Get the macro averaging function
        macro_averaging_function = _average_methods("macro")

        # Compute the result using the macro averaging function
        result = macro_averaging_function(self.preds, self.labels, self.simple_metric)

        # Assert the result is as expected
        assert torch.allclose(result, expected_macro_average, 1e-4)

    # The function returns a callable that computes metrics correctly for "none" averaging
    def test_none_averaging(self):
        # Expected output for "none" averaging
        expected_output = torch.tensor([0.5000, 0.6667, 0.5000])

        none_averaging_func = _average_methods("none")

        # Call the function
        result = none_averaging_func(self.preds, self.labels, self.simple_metric)

        # Assert the result is as expected
        assert torch.allclose(result, expected_output, 1e-4)


class Test_PrepareTensors:

    # Apply sigmoid to predictions outside [0, 1] range
    def test_apply_sigmoid_outside_range(self):

        # Define prediction tensor with values outside [0, 1]
        prediction = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        label = torch.tensor([[1, 0], [0, 1]])

        # Expected output after applying sigmoid and thresholding
        expected_prediction = sigmoid(prediction) > 0.5

        # Call the function
        result_prediction, result_label = _prepare_tensors(prediction, label)

        # Assert the result is as expected
        assert torch.allclose(
            torch.tensor(result_prediction), expected_prediction, 1e-4
        )
        assert torch.allclose(torch.tensor(result_label), label, 1e-4)

    # Handle predictions with all values exactly 0 or 1
    def test_handle_predictions_all_zero_or_one(self):

        # Define prediction tensor with all values exactly 0 or 1
        prediction = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        label = torch.tensor([[1, 0], [0, 1]])

        # Expected output after thresholding (no sigmoid applied)
        expected_prediction = prediction > 0.5

        # Call the function
        result_prediction, result_label = _prepare_tensors(prediction, label)

        # Assert the result is as expected
        assert torch.allclose(
            torch.tensor(result_prediction), expected_prediction, 1e-4
        )
        assert torch.allclose(torch.tensor(result_label), label, 1e-4)


class TestDice:

    # micro, macro, weighted, none
    def test_dice_coefficient_with_various_averaging_methods(self):
        prediction, label = get_3d_predictions()
        # Define expected results for different averaging methods
        expected_results = {
            "micro": torch.tensor(0.5872),
            "macro": torch.tensor(0.5836),
            "weighted": torch.tensor(0.5870),
            "none": torch.tensor([0.5373, 0.6500, 0.5634]),
        }

        # Test the dice function with different averaging methods
        for avg_method, expected in expected_results.items():
            result = dice(prediction, label, average=avg_method)  # type: ignore
            assert torch.allclose(result, expected, atol=1e-4)


class TestSurfaceDice:

    # micro, macro, weighted, and none
    def test_surface_dice_with_various_averaging_methods(self, mocker):
        prediction, label = get_3d_predictions()

        expected = {
            "macro": torch.tensor(0.9307, dtype=torch.float64),
            "none": torch.tensor([0.9432, 0.9278, 0.9211], dtype=torch.float64),
        }

        # Test for different averaging methods
        for average in ["macro", "none"]:
            result = surface_dice(prediction, label, average=average, tolerance=0.5)
            assert isinstance(result, torch.Tensor)
            assert torch.allclose(result, expected[average], atol=1e-4)


class TestHausdorffDistance:
    # Supports "micro", "macro", and "none" averaging methods
    def test_hausdorff_distance_with_various_averaging_methods(self):
        prediction, label = get_3d_predictions()
        # Define expected results for different averaging methods
        expected_results = {
            "micro": torch.tensor(1.4142135623730951, dtype=torch.float64),
            "macro": torch.tensor(1.4142, dtype=torch.float64),
            "none": torch.tensor([1.4142, 1.4142, 1.4142], dtype=torch.float64),
        }

        # Test the hausdorff_distance function with different averaging methods
        for avg_method, expected in expected_results.items():
            result = hausdorff_distance(prediction, label, average=avg_method)  # type: ignore
            assert torch.allclose(result, expected, atol=1e-5)


class TestGeneralisedEnergyDistance:

    # Supports "micro", "macro", and "none" averaging methods
    def test_generalised_energy_distance_averaging_methods(self):
        np.random.seed(69)
        a = torch.tensor(np.random.rand(5, 3, 4, 6, 3))
        np.random.seed(69)
        b = torch.tensor(np.random.randint(0, 2, (5, 3, 4, 6, 3)))

        # Expected results for different averaging methods
        expected_results = {
            "micro": torch.tensor(0.5734),
            "macro": torch.tensor(0.576),
            "none": torch.tensor([0.5618, 0.6093, 0.5552]),
        }

        # Test the generalised_energy_distance function with different averaging methods
        for avg_method, expected in expected_results.items():
            result = generalised_energy_distance(a, b, average=avg_method)  # type: ignore
            assert torch.allclose(result, expected, atol=1e-4)


class TestRcCurveStats:

    # Correctly computes coverage, selective risks, and weights for valid inputs
    def test_valid_inputs(self):
        risks = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        confids = torch.tensor([0.8, 0.4, 0.4, 0.6, 0.1])

        expected_coverages = [1.0, 0.8, 0.6, 0.2]
        expected_selective_risks = [
            torch.tensor(0.3000),
            torch.tensor(0.2500),
            torch.tensor(0.2667),
            torch.tensor(0.1000),
        ]
        expected_weights = [0.2, 0.2, 0.4]

        coverages, selective_risks, weights = rc_curve_stats(risks, confids)

        assert np.allclose(coverages, expected_coverages)
        assert torch.allclose(
            torch.tensor(selective_risks),
            torch.tensor(expected_selective_risks),
            atol=1e-3,
        )
        assert np.allclose(weights, expected_weights)


class TestAURC:

    # Computes AURC correctly for valid risk and confidence tensors
    def test_aurc_computation_valid_tensors(self):
        risks = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        confids = torch.tensor([0.8, 0.4, 0.4, 0.6, 0.1])

        expected_aurc = torch.tensor(0.1800)
        expected_coverage = [1.0, 0.8, 0.6, 0.2]
        expected_selective_risks = [
            torch.tensor(0.3000),
            torch.tensor(0.2500),
            torch.tensor(0.2667),
            torch.tensor(0.1000),
        ]
        aurc_value, coverage, selective_risks = aurc(risks, confids)
        assert torch.isclose(aurc_value, expected_aurc)
        assert coverage == expected_coverage
        assert torch.allclose(
            torch.tensor(selective_risks),
            torch.tensor(expected_selective_risks),
            atol=1e-3,
        )


class TestEAURC:

    # Computes EAURC correctly for valid risk and confidence tensors
    def test_eaurc_computation_valid_tensors(self):
        risks = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        confids = torch.tensor([0.8, 0.4, 0.4, 0.6, 0.1])

        expected_eaurc = torch.tensor(-0.0200)
        expected_coverage = [1.0, 0.8, 0.6, 0.2]
        expected_selective_risks = [
            torch.tensor(0.3000),
            torch.tensor(0.2500),
            torch.tensor(0.2667),
            torch.tensor(0.1000),
        ]
        eaurc_value, coverage, selective_risks = eaurc(risks, confids)
        assert torch.isclose(eaurc_value, expected_eaurc)
        assert coverage == expected_coverage
        assert torch.allclose(
            torch.tensor(selective_risks),
            torch.tensor(expected_selective_risks),
            atol=1e-3,
        )
