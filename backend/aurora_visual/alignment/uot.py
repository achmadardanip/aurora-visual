"""Differentiable log-domain generalized Sinkhorn, KL reference a outer b."""

from dataclasses import dataclass

import torch


@dataclass
class Transport:
    coupling: torch.Tensor
    unmatched_mass: torch.Tensor
    diagnostics: dict


def generalized_kl(p, q):
    return (torch.where(p > 0, p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log()), 0) - p + q).sum()


def objective(plan, cost, a, b, epsilon=0.08, tau_t=0.8, tau_v=0.8):
    return (
        (plan * cost).sum()
        + epsilon * generalized_kl(plan, a[:, None] * b[None, :])
        + tau_t * generalized_kl(plan.sum(1), a)
        + tau_v * generalized_kl(plan.sum(0), b)
    )


def sinkhorn(
    cost, a=None, b=None, epsilon=0.08, tau_t=0.8, tau_v=0.8, max_iter=300, tolerance=1e-7, balanced=False
):
    if cost.ndim != 2 or not cost.numel() or not torch.isfinite(cost).all():
        raise ValueError("Cost matrix must be nonempty and finite")
    if epsilon <= 0 or tau_t <= 0 or tau_v <= 0 or max_iter < 1 or tolerance <= 0:
        raise ValueError("Invalid transport configuration")
    n, m = cost.shape
    a = torch.full((n,), 1 / n, dtype=cost.dtype, device=cost.device) if a is None else a
    b = torch.full((m,), 1 / m, dtype=cost.dtype, device=cost.device) if b is None else b
    if (
        a.shape != (n,)
        or b.shape != (m,)
        or not torch.isfinite(a).all()
        or not torch.isfinite(b).all()
        or (a <= 0).any()
        or (b <= 0).any()
    ):
        raise ValueError("Transport priors must be positive and finite")
    if balanced and not torch.isclose(a.sum(), b.sum()):
        raise ValueError("Balanced transport requires equal total mass")
    la, lb = a.log(), b.log()
    kernel = la[:, None] + lb[None, :] - cost / epsilon
    u, v = torch.zeros_like(a), torch.zeros_like(b)
    pt = 1 if balanced else tau_t / (tau_t + epsilon)
    pv = 1 if balanced else tau_v / (tau_v + epsilon)
    converged, error = False, float("inf")
    for iteration in range(1, max_iter + 1):
        new_u = pt * (la - torch.logsumexp(kernel + v[None, :], dim=1))
        new_v = pv * (lb - torch.logsumexp(kernel + new_u[:, None], dim=0))
        error = float(torch.maximum((new_u - u).abs().max(), (new_v - v).abs().max()).detach())
        u, v = new_u, new_v
        if error < tolerance:
            converged = True
            break
    plan = torch.exp(kernel + u[:, None] + v[None, :])
    if not torch.isfinite(plan).all():
        raise ValueError("Non-finite coupling")
    unmatched = (a - plan.sum(1)).clamp_min(0) / a.clamp_min(1e-12)
    return Transport(
        plan,
        unmatched,
        {
            "algorithm": "balanced-ot" if balanced else "uot",
            "converged": converged,
            "iterations": iteration,
            "fixed_point_error": error,
            "transported_mass": float(plan.sum().detach()),
            "row_l1": float((plan.sum(1) - a).abs().sum().detach()),
            "column_l1": float((plan.sum(0) - b).abs().sum().detach()),
            "objective": float(objective(plan, cost, a, b, epsilon, tau_t, tau_v).detach()),
            "epsilon": epsilon,
            "tau_t": tau_t,
            "tau_v": tau_v,
        },
    )


def align(cost, method="uot", **kwargs):
    if method in ("uot", "balanced-ot"):
        return sinkhorn(cost, balanced=method == "balanced-ot", **kwargs)
    if cost.ndim != 2 or not cost.numel() or not torch.isfinite(cost).all():
        raise ValueError("Empty or non-finite region cost")
    n, m = cost.shape
    if method == "max-region":
        coupling = torch.nn.functional.one_hot(cost.argmin(1), m).to(cost.dtype) / n
    elif method in ("mean-region", "global"):
        coupling = torch.ones_like(cost) / (n * m)
    elif method == "attention":
        coupling = torch.softmax(-cost / kwargs.get("epsilon", 0.08), dim=1) / n
    else:
        raise ValueError("Unknown alignment method")
    return Transport(
        coupling,
        torch.zeros(n, device=cost.device),
        {"algorithm": method, "converged": True, "transported_mass": 1.0},
    )
