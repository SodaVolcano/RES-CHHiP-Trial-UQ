import numpy as np
import torch

from ..context import metrics

rc_curve_stats = metrics.rc_curve_stats
aurc = metrics.aurc
eaurc = metrics.eaurc


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
