"""Trainable fusion head: multinomial logistic regression (CPU, numpy).

A random/untouched head is NEVER presented as a live prediction — predict()
raises unless the checkpoint carries trained metadata. Training/calibration
smoke runs on labeled fixtures to prove the computation path.
"""

import json
from pathlib import Path

import numpy as np

from aurora_decision.core.fusion import FACT_LABELS, FEATURE_NAMES


class HeadNotTrained(Exception):
    pass


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class FusionHead:
    """Softmax regression over fusion features -> S/C/IE probabilities."""

    def __init__(self, weights=None, bias=None, metadata=None):
        self.weights = weights  # (F, C)
        self.bias = bias  # (C,)
        self.metadata = metadata or {}

    @property
    def trained(self) -> bool:
        return self.weights is not None and bool(self.metadata.get("trained"))

    @classmethod
    def train(
        cls,
        x: np.ndarray,
        y: list[str],
        epochs: int = 300,
        lr: float = 0.05,
        seed: int = 17,
        l2: float = 1e-3,
    ) -> "FusionHead":
        """Class-balanced softmax regression with L2; labels are S/C/IE names."""
        rng = np.random.default_rng(seed)
        n, f = x.shape
        counts = {label: max(1, y.count(label)) for label in FACT_LABELS}
        sample_weight = np.array([n / (len(y) * counts[label]) for label in y])
        labels = np.array([FACT_LABELS.index(label) for label in y])
        w = rng.normal(0, 0.01, (f, len(FACT_LABELS)))
        b = np.zeros(len(FACT_LABELS))
        for _ in range(epochs):
            probabilities = softmax(x @ w + b)
            indices = np.arange(n)
            grad_z = probabilities.copy()
            grad_z[indices, labels] -= 1.0
            grad_z *= sample_weight[:, None]
            grad_w = x.T @ grad_z / n + l2 * w
            grad_b = grad_z.mean(axis=0)
            w, b = w - lr * grad_w, b - lr * grad_b
        head = cls(
            w,
            b,
            {
                "trained": True,
                "feature_names": FEATURE_NAMES,
                "label_map": list(FACT_LABELS),
                "epochs": epochs,
                "seed": seed,
            },
        )
        return head

    def predict(self, x: np.ndarray) -> list[dict]:
        if not self.trained:
            raise HeadNotTrained("Head belum dilatih; prediksi nyata tidak tersedia.")
        probabilities = softmax(np.atleast_2d(x) @ self.weights + self.bias)
        return [{label: float(p) for label, p in zip(FACT_LABELS, row)} for row in probabilities]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, allow_nan=False))
        return path

    @classmethod
    def load(cls, path: Path) -> "FusionHead":
        payload = json.loads(Path(path).read_text())
        return cls(
            np.array(payload["weights"]),
            np.array(payload["bias"]),
            payload.get("metadata", {}),
        )
