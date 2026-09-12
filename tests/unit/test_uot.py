import numpy as np
import pytest
import torch
from aurora_visual.alignment.model import VisualHead
from aurora_visual.alignment.uot import align, generalized_kl, objective, sinkhorn
from aurora_visual.training.data import smoke_samples
from scipy.optimize import minimize


def test_uot_reference_objective():
    cost = torch.tensor([[0.05, 0.9], [0.7, 0.2]], dtype=torch.float64)
    a = torch.tensor([0.4, 0.6], dtype=torch.float64)
    b = torch.tensor([0.55, 0.45], dtype=torch.float64)
    result = sinkhorn(cost, a, b, epsilon=0.15, tau_t=0.7, tau_v=0.8, max_iter=2000, tolerance=1e-12)

    def ref(flat):
        return float(objective(torch.tensor(flat.reshape(2, 2)), cost, a, b, 0.15, 0.7, 0.8))

    numerical = minimize(
        ref,
        np.full(4, 0.2),
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * 4,
        options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 2000},
    )
    assert numerical.success
    assert abs(result.diagnostics["objective"] - numerical.fun) < 1e-7
    assert np.allclose(result.coupling.numpy(), numerical.x.reshape(2, 2), atol=2e-5)
    assert result.diagnostics["converged"]


def test_mass_nonnegative_range_and_difference():
    cost = torch.full((3, 4), 2.0, dtype=torch.float64)
    u = sinkhorn(cost)
    b = sinkhorn(cost, balanced=True)
    assert torch.isfinite(u.coupling).all() and (u.coupling >= 0).all()
    assert ((u.unmatched_mass >= 0) & (u.unmatched_mass <= 1)).all()
    assert u.coupling.sum() < b.coupling.sum() * 0.6
    assert torch.allclose(b.coupling.sum(1), torch.full((3,), 1 / 3, dtype=torch.float64))


def test_large_mass_penalty_balanced_limit():
    cost = torch.tensor([[0.1, 0.7], [0.3, 0.2]], dtype=torch.float64)
    balanced = sinkhorn(cost, balanced=True, max_iter=5000, tolerance=1e-12)
    u = sinkhorn(cost, tau_t=1e4, tau_v=1e4, max_iter=8000, tolerance=1e-9)
    assert torch.allclose(u.coupling, balanced.coupling, atol=1e-4)


def test_kl_includes_mass_terms():
    assert float(generalized_kl(torch.tensor([0.0]), torch.tensor([2.0]))) == 2


@pytest.mark.parametrize(
    "cost", [torch.empty(2, 0), torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]])]
)
def test_bad_matrix(cost):
    with pytest.raises(ValueError):
        sinkhorn(cost)


def test_nonconvergence_reported():
    result = sinkhorn(torch.tensor([[0.2, 8.0], [0.1, 0.3]]), max_iter=1, tolerance=1e-14)
    assert not result.diagnostics["converged"]


def test_autograd_projector_and_learned_cost():
    torch.manual_seed(4)
    model = VisualHead()
    result = model(**smoke_samples()[0]["inputs"])
    (result["logits"].square().mean() + result["transport"].coupling.square().sum()).backward()
    for p in (model.text_projector.weight, model.region_projector.weight, model.weights):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.parametrize("method", ["global", "max-region", "mean-region", "attention", "balanced-ot", "uot"])
def test_all_baselines(method):
    result = align(torch.tensor([[0.2, 0.8], [0.5, 0.3]]), method)
    assert result.coupling.shape == (2, 2) and torch.isfinite(result.coupling).all()
