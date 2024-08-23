import torch
from torch import nn

from uncertainty.models.ensemble import DeepEnsemble
from .context import uncertainty

DeepEnsemble = uncertainty.models.ensemble.DeepEnsemble


class TestDeepEnsemble:
    # Initializes ensemble with a callable model and ensemble size
    def test_initializes_with_callable_model_and_ensemble_size(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        ensemble_size = 3
        model = SimpleModel
        ensemble = DeepEnsemble(model, ensemble_size)

        assert len(ensemble.models) == ensemble_size
        assert all(isinstance(m, SimpleModel) for m in ensemble.models)

    # Initializes ensemble with a list of models
    def test_initializes_with_list_of_models(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        ensemble_size = 3
        models_list = [SimpleModel() for _ in range(ensemble_size)]
        ensemble = DeepEnsemble(models_list, ensemble_size)  # type: ignore

        assert len(ensemble.models) == ensemble_size
        assert all(isinstance(m, SimpleModel) for m in ensemble.models)

    # Forwards input tensor through all models in the ensemble
    def test_forwards_input_through_all_models(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        ensemble_size = 3
        model = SimpleModel
        ensemble = DeepEnsemble(model, ensemble_size)

        input_tensor = torch.randn(1, 10)
        outputs = ensemble.forward(input_tensor)

        assert len(outputs) == ensemble_size
        assert all(isinstance(o, torch.Tensor) for o in outputs)
        assert not all(torch.equal(outputs[0], o) for o in outputs[1:])
