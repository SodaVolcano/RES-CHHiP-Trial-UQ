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
    prediction = torch.tensor(
        [
            [
                [
                    [0.8398, 0.8042, 0.1213],
                    [0.5309, 0.6646, 0.4077],
                    [0.0888, 0.2429, 0.7053],
                    [0.6216, 0.9188, 0.0185],
                    [0.8741, 0.0560, 0.9659],
                    [0.0073, 0.3628, 0.4197],
                ],
                [
                    [0.6444, 0.0099, 0.5925],
                    [0.9631, 0.6958, 0.9157],
                    [0.5523, 0.2344, 0.3262],
                    [0.4521, 0.7020, 0.6274],
                    [0.0945, 0.8525, 0.3572],
                    [0.7492, 0.3579, 0.5453],
                ],
                [
                    [0.8171, 0.0570, 0.4560],
                    [0.0183, 0.5854, 0.6620],
                    [0.0158, 0.5309, 0.9056],
                    [0.6011, 0.6072, 0.5147],
                    [0.7654, 0.5434, 0.3774],
                    [0.3056, 0.6771, 0.3802],
                ],
                [
                    [0.2426, 0.8268, 0.8742],
                    [0.6367, 0.3849, 0.0412],
                    [0.8489, 0.7989, 0.9141],
                    [0.1347, 0.1917, 0.6456],
                    [0.1305, 0.8144, 0.8384],
                    [0.8903, 0.0498, 0.1710],
                ],
            ],
            [
                [
                    [0.8857, 0.8745, 0.7445],
                    [0.3265, 0.9798, 0.7475],
                    [0.9911, 0.4404, 0.1755],
                    [0.8882, 0.2830, 0.8929],
                    [0.5384, 0.0219, 0.0473],
                    [0.8484, 0.8749, 0.1102],
                ],
                [
                    [0.4257, 0.3868, 0.1889],
                    [0.4410, 0.0286, 0.5514],
                    [0.8156, 0.5011, 0.7597],
                    [0.6858, 0.3605, 0.3658],
                    [0.9362, 0.0408, 0.6278],
                    [0.1817, 0.7110, 0.2446],
                ],
                [
                    [0.1089, 0.3925, 0.9651],
                    [0.8191, 0.8030, 0.2029],
                    [0.2073, 0.0954, 0.4212],
                    [0.4838, 0.3973, 0.4797],
                    [0.7069, 0.1273, 0.1206],
                    [0.6707, 0.4570, 0.1266],
                ],
                [
                    [0.1082, 0.3762, 0.5488],
                    [0.5367, 0.1702, 0.9320],
                    [0.0241, 0.5887, 0.8592],
                    [0.3443, 0.4129, 0.5583],
                    [0.0072, 0.1850, 0.4954],
                    [0.4377, 0.8669, 0.3738],
                ],
            ],
            [
                [
                    [0.5977, 0.4277, 0.7407],
                    [0.0071, 0.4448, 0.8276],
                    [0.8949, 0.6671, 0.7350],
                    [0.5121, 0.8455, 0.4126],
                    [0.5557, 0.1370, 0.7244],
                    [0.7074, 0.4282, 0.2949],
                ],
                [
                    [0.7710, 0.2102, 0.1641],
                    [0.5880, 0.1561, 0.7975],
                    [0.0201, 0.0893, 0.0379],
                    [0.0911, 0.6519, 0.0645],
                    [0.2580, 0.0427, 0.9386],
                    [0.2387, 0.1695, 0.3943],
                ],
                [
                    [0.3687, 0.4875, 0.6619],
                    [0.4345, 0.0208, 0.8953],
                    [0.1165, 0.4610, 0.9498],
                    [0.4855, 0.7458, 0.8383],
                    [0.4455, 0.4383, 0.4106],
                    [0.5997, 0.0772, 0.2856],
                ],
                [
                    [0.0806, 0.2498, 0.8602],
                    [0.4760, 0.3300, 0.5000],
                    [0.5835, 0.4999, 0.6664],
                    [0.4822, 0.5441, 0.3877],
                    [0.0354, 0.2914, 0.1893],
                    [0.7001, 0.4933, 0.5919],
                ],
            ],
        ]
    )
    label = torch.tensor(
        [
            [
                [[0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 0], [0, 1, 0]],
                [[0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]],
                [[1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]],
                [[0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1]],
            ],
            [
                [[0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 0]],
                [[1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 0]],
                [[1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1]],
                [[0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 0, 0]],
            ],
            [
                [[1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1]],
                [[0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0]],
                [[1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 0], [0, 1, 1]],
            ],
        ]
    )
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
            "micro": torch.tensor(0.4793),
            "macro": torch.tensor(0.4779),
            "weighted": torch.tensor(0.4770),
            "none": torch.tensor([0.5128, 0.4595, 0.4615]),
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
            "macro": torch.tensor(0.9308, dtype=torch.float64),
            "none": torch.tensor([0.9687, 0.9079, 0.9157], dtype=torch.float64),
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
            "micro": torch.tensor(1.4142, dtype=torch.float64),
            "macro": torch.tensor(1.6095, dtype=torch.float64),
            "none": torch.tensor([1.4142, 1.4142, 2.000], dtype=torch.float64),
        }

        # Test the hausdorff_distance function with different averaging methods
        for avg_method, expected in expected_results.items():
            result = hausdorff_distance(prediction, label, average=avg_method)  # type: ignore
            assert torch.allclose(result, expected, atol=1e-4)


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
