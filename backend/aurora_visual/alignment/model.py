import torch
from torch import nn
from torch.nn import functional as F

from aurora_visual.alignment.uot import align

ROLES = ["actor", "action", "object", "attribute", "location", "time", "quantity", "relation", "cause"]
LABELS = ["Supported", "Contradicted", "Unobservable"]


class VisualHead(nn.Module):
    """Unary geometry approximation to relational cost; not graph optimal transport."""

    def __init__(self, embedding_dim=12, hidden=32, use_unmatched=True, cosine_only=False):
        super().__init__()
        self.embedding_dim, self.hidden = embedding_dim, hidden
        self.use_unmatched, self.cosine_only = use_unmatched, cosine_only
        self.text_projector = nn.Linear(embedding_dim + 13, hidden)
        self.region_projector = nn.Linear(embedding_dim * 2 + 5, hidden)
        self.role_projector = nn.Linear(9, hidden, bias=False)
        self.weights = nn.Parameter(torch.zeros(4))
        self.head = nn.Sequential(nn.Linear(hidden * 2 + 6, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(
        self,
        text,
        regions,
        atom_meta,
        geometry,
        global_feature,
        relation_targets=None,
        relation_mask=None,
        global_text=None,
        explicit_counter=None,
        method="uot",
    ):
        if method == "global":
            if global_text is None:
                raise ValueError("Global baseline requires full-caption embedding")
            text = global_text.expand_as(text)
            atom_meta = torch.zeros_like(atom_meta)
            regions = global_feature.expand_as(regions)
            geometry = torch.zeros_like(geometry)
            relation_targets, explicit_counter = None, None
        h_t = F.normalize(self.text_projector(torch.cat([text, atom_meta], -1)), dim=-1)
        gv = global_feature.expand(regions.shape[0], -1)
        h_v = F.normalize(self.region_projector(torch.cat([regions, gv, geometry], -1)), dim=-1)
        cosine = (1 - h_t @ h_v.T).clamp_min(0)
        role = 1 - F.normalize(self.role_projector(atom_meta[:, :9]), dim=-1) @ h_v.T
        attr = torch.cdist(text, regions).square() / max(1, text.shape[-1])
        # Targets are expected relative centers from linked atom/dependency grounding.
        centers = (geometry[:, :2] + geometry[:, 2:4]) / 2
        relation = (
            torch.zeros_like(cosine)
            if relation_targets is None
            else torch.cdist(relation_targets, centers).square()
        )
        if relation_mask is not None:
            relation = relation * relation_mask[:, None]
        components = torch.stack([cosine, role.clamp_min(0), attr, relation])
        cost = cosine if self.cosine_only else (self.weights.softmax(0)[:, None, None] * components).sum(0)
        transport = align(cost, method=method, tolerance=1e-5, max_iter=200)
        plan = transport.coupling
        normalized = plan / plan.sum(1, keepdim=True).clamp_min(1e-8)
        pooled = normalized @ h_v
        entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(1)
        counter = torch.zeros_like(entropy) if explicit_counter is None else explicit_counter
        rho = transport.unmatched_mass if self.use_unmatched else torch.zeros_like(entropy)
        extras = torch.stack(
            [cost.min(1).values, entropy, rho, atom_meta[:, -1], normalized @ geometry[:, -1], counter], -1
        )
        logits = self.head(torch.cat([h_t, pooled, extras], -1))
        return {
            "logits": logits,
            "probabilities": logits.softmax(-1),
            "transport": transport,
            "cost": cost,
            "atom_features": h_t,
            "region_features": h_v,
            "entropy": entropy,
        }
