import torch
from ..context import evaluation


evaluate_prediction = evaluation.evaluate_prediction
evaluate_predictions = evaluation.evaluate_predictions


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


METRIC_NAMES = [
    "hd",
    "hd95",
    "asd",
    "assd",
    "dice",
    "surface_dice_0.5",
    "recall",
    "precision",
    "mean_variance",
    "mean_entropy",
]


class TestEvaluatePrediction:

    # Evaluate prediction with single metric and macro averaging returns correct tensor
    def test_single_metric_macro_averaging(self):
        prediction, label = get_3d_predictions()
        metric_names = ["dice"]

        result = evaluate_prediction(prediction, label, metric_names, average="macro")

        expected = torch.tensor([0.4779])
        assert torch.allclose(result, expected, atol=1e-4)

    # Micro averaging calculates metrics globally across all classes
    def test_micro_averaging_global_metrics(self):
        prediction, label = get_3d_predictions()

        # Call the evaluate_prediction function with micro averaging
        result = evaluate_prediction(prediction, label, METRIC_NAMES, average="micro")

        expected_result = torch.tensor(
            [
                1.4142,
                1.0000,
                0.4800,
                0.5265,
                0.4793,
                torch.nan,  # surface Dice don't work with micro averaging
                0.4444,
                0.5200,
                0.1676,
                0.5015,
            ],
            dtype=torch.float64,
        )

        result = torch.nan_to_num(result, nan=0.0)  # replace nan with 0
        expected_result = torch.nan_to_num(expected_result, nan=0.0)

        # Assert the result is as expected
        assert torch.allclose(result, expected_result, atol=1e-4)

    # Multiple metrics evaluation with macro averaging concatenates results correctly
    def test_evaluate_prediction_macro_averaging(self):
        prediction, label = get_3d_predictions()
        # Expected result for macro averaging
        expected_result = torch.tensor(
            [
                1.6095,
                1.0000,
                0.4760,
                0.5367,
                0.4779,
                0.9308,
                0.4471,
                0.5240,
                0.1676,
                0.5015,
            ],
            dtype=torch.float64,
        )

        # Call the evaluate_prediction function
        result = evaluate_prediction(prediction, label, METRIC_NAMES, average="macro")

        # Assert the result is as expected
        assert torch.allclose(result, expected_result, atol=1e-4)

    # No averaging (average="none") returns per-class metrics
    def test_no_averaging_returns_per_class_metrics(self):
        prediction, label = get_3d_predictions()

        # Expected output for "none" averaging
        expected_output = torch.tensor(
            [
                1.4142,
                1.4142,
                2.0000,
                1.0000,
                1.0000,
                1.0000,
                0.5122,
                0.4516,
                0.4643,
                0.4925,
                0.5573,
                0.5602,
                0.5128,
                0.4595,
                0.4615,
                0.9687,
                0.9079,
                0.9157,
                0.5405,
                0.3953,
                0.4054,
                0.4878,
                0.5484,
                0.5357,
                0.1622,
                0.1636,
                0.1770,
                0.4882,
                0.4919,
                0.5244,
            ],
            dtype=torch.float64,
        )

        # Call the evaluate_prediction function with average="none"
        result = evaluate_prediction(prediction, label, METRIC_NAMES, average="none")

        # Assert the result is as expected
        assert torch.allclose(result, expected_output, atol=1e-4)


class TestEvaluatePredictions:

    # Evaluate single metric on single prediction tensor against label with macro averaging
    def test_single_metric_various_averaging(self):
        # Create test data
        prediction, label = get_3d_predictions()
        predictions = prediction.expand(20, -1, -1, -1, -1)
        metric_names = ["dice"]

        # Call function under test
        macro = evaluate_predictions(predictions, label, metric_names, average="macro")
        micro = evaluate_predictions(predictions, label, metric_names, average="micro")
        none = evaluate_predictions(predictions, label, metric_names, average="none")

        assert torch.allclose(macro, torch.tensor([0.4779]), atol=1e-4)
        assert torch.allclose(micro, torch.tensor([0.4793]), atol=1e-4)
        assert torch.allclose(none, torch.tensor([0.5128, 0.4595, 0.4615]), atol=1e-4)

    # Evaluate metrics with aggregate_before_eval=True
    def test_all_metrics_aggregation(self):
        _, label = get_3d_predictions()
        # pairwise metric will remove duplicates so we need to ensure each of
        # 20 predictions are unique using rand() instead of expand()
        torch.manual_seed(42)
        predictions = torch.rand(20, 3, 4, 6, 3)

        # Define metric names to evaluate
        metric_names = METRIC_NAMES + ["pairwise_dice", "pairwise_surface_dice_0.2"]

        expected = {
            "macro": torch.tensor(
                [
                    1.2761,
                    1.0000,
                    0.4616,
                    0.5001,
                    0.5054,
                    0.9322,
                    0.4806,
                    0.5449,
                    0.0821,
                    5.0526,
                    0.4958,
                    0.9322,
                ],
                dtype=torch.float64,
            ),
            "micro": torch.tensor(
                [
                    1.4142,
                    1.0000,
                    0.4615,
                    0.4951,
                    0.5068,
                    torch.nan,
                    0.4786,
                    0.5385,
                    0.0821,
                    5.0526,
                    0.4985,
                    torch.nan,
                ],
                dtype=torch.float64,
            ),
            "none": torch.tensor(
                [
                    1.4142,
                    1.0000,
                    1.4142,
                    1.0000,
                    1.0000,
                    1.0000,
                    0.4667,
                    0.3871,
                    0.5309,
                    0.5286,
                    0.4865,
                    0.4854,
                    0.4776,
                    0.5135,
                    0.5250,
                    0.9101,
                    0.9335,
                    0.9529,
                    0.4324,
                    0.4419,
                    0.5676,
                    0.5333,
                    0.6129,
                    0.4884,
                    0.0791,
                    0.0854,
                    0.0818,
                    5.1266,
                    5.0141,
                    5.0170,
                    0.4936,
                    0.4761,
                    0.5177,
                    0.9101,
                    0.9335,
                    0.9529,
                ],
                dtype=torch.float64,
            ),
        }

        # Assert the result is as expected
        for avg in expected.keys():
            result = evaluate_predictions(predictions, label, metric_names, average=avg)
            # repalce nans with zero
            result = torch.nan_to_num(result, nan=0.0)
            expected[avg] = torch.nan_to_num(expected[avg], nan=0.0)
            assert torch.allclose(result, expected[avg], atol=1e-4)

    # Evaluate spatial metrics with aggregate_before_eval=False and average results
    def test_metrics_aggregate_false_average_results(self):
        _, label = get_3d_predictions()
        # pairwise metric will remove duplicates so we need to ensure each of
        # 20 predictions are unique using rand() instead of expand()
        torch.manual_seed(42)
        predictions = torch.rand(20, 3, 4, 6, 3)

        # Define metric names to evaluate
        metric_names = METRIC_NAMES + ["pairwise_dice", "pairwise_surface_dice_0.2"]

        expected = {
            "macro": torch.tensor(
                [
                    1.2607,
                    1.0173,
                    0.4564,
                    0.4847,
                    0.5232,
                    0.9378,
                    0.5053,
                    0.5490,
                    0.0821,
                    5.0526,
                    0.4958,
                    0.9378,
                ],
                dtype=torch.float64,
            ),
            "micro": torch.tensor(
                [
                    1.2278,
                    1.0000,
                    0.4513,
                    0.4763,
                    0.5252,
                    torch.nan,
                    0.5047,
                    0.5487,
                    0.0821,
                    5.0526,
                    0.4985,
                    torch.nan,
                ],
                dtype=torch.float64,
            ),
            "none": torch.tensor(
                [
                    1.1657,
                    1.2230,
                    1.3935,
                    1.0000,
                    1.0124,
                    1.0394,
                    0.4735,
                    0.3881,
                    0.5076,
                    0.4881,
                    0.4635,
                    0.5026,
                    0.5156,
                    0.5447,
                    0.5094,
                    0.9507,
                    0.9203,
                    0.9423,
                    0.5095,
                    0.4942,
                    0.5122,
                    0.5259,
                    0.6123,
                    0.5088,
                    0.0791,
                    0.0854,
                    0.0818,
                    5.1266,
                    5.0141,
                    5.0170,
                    0.4936,
                    0.4761,
                    0.5177,
                    0.9507,
                    0.9203,
                    0.9423,
                ],
                dtype=torch.float64,
            ),
        }

        # Assert the result is as expected
        for avg in expected.keys():
            result = evaluate_predictions(
                predictions,
                label,
                metric_names,
                average=avg,
                aggregate_before_eval=False,
            )
            # repalce nans with zero
            result = torch.nan_to_num(result, nan=0.0)
            expected[avg] = torch.nan_to_num(expected[avg], nan=0.0)
            assert torch.allclose(result, expected[avg], atol=1e-4)
