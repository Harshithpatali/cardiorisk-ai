import numpy as np
import torch

from explainability import explain_observation, explanation_diagnostics


def test_integrated_gradients_completeness_for_linear_model():
    model = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -3.0]]))
        model.bias.fill_(0.25)
    background = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    observation = np.array([[0.8, 0.2]], dtype=np.float32)
    attributions = explain_observation(model, background, observation, steps=16)
    diagnostics = explanation_diagnostics(
        model, background, observation, attributions
    )
    assert diagnostics["absolute_residual"] < 1e-5
    assert diagnostics["relative_residual"] < 1e-5
