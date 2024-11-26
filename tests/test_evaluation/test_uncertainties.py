import torch
from ..context import evaluation
from torchmetrics.classification import MultilabelF1Score


_aggregate_map_metric = evaluation.uncertainties._aggregate_map_metric
_pairwise_metric = evaluation.uncertainties._pairwise_metric
dice = evaluation.dice


class Test_AggregateMapMetric:

    # test variance map with various averaging methods
    def test_aggregate_map_metric_with_various_averaging_methods(self):
        torch.random.manual_seed(42)
        map_metric = torch.rand(5, 3, 10, 10, 15).var(dim=0)

        # Expected results for different averaging methods
        expected_results = {
            "micro": torch.tensor(0.0831),
            "macro": torch.tensor(0.0831),
            "none": torch.tensor([0.0847, 0.0813, 0.0832]),
        }

        # Test the _aggregate_map_metric function with different averaging methods
        for avg_method, expected in expected_results.items():
            result = _aggregate_map_metric(map_metric, average=avg_method)  # type: ignore
            assert torch.allclose(result, expected, atol=1e-4)


class Test_PairwiseMetric:

    # test works for all averaging methods with aggregation
    def test_pairwise_metric_with_averaging_methods(self):
        torch.random.manual_seed(42)
        predictions = torch.rand(5, 3, 10, 10, 15) > 0.5

        # Expected results for different averaging methods
        expected_results = {
            "micro": torch.tensor(0.4990),
            "macro": torch.tensor(0.4989),
            "none": torch.tensor([0.4955, 0.5030, 0.4981]),
        }

        # Test the _pairwise_metric function with different averaging methods
        for avg_method, expected in expected_results.items():
            result = _pairwise_metric(
                dice,
                predictions,
                average=avg_method,  # type: ignore
            )
            assert torch.allclose(result, expected, atol=1e-4)
