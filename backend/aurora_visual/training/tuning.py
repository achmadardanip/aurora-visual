"""Optuna hyperparameter tuning for the three-state head.

Objective: validation macro-F1 (atom level) with event-grouped data. Tuning on
fixtures is an engineering smoke only; research tuning must use licensed
research manifests and is never run on the test split (docs/evaluation.md).
"""

from pathlib import Path

import optuna

from aurora_visual.training.data import dataset_hash
from aurora_visual.training.engine import train

SEARCH_SPACE = {
    "learning_rate": {"low": 1e-4, "high": 1e-2, "log": True},
    "hidden": [16, 32, 64],
    "method": ["uot", "balanced-ot", "attention", "max-region"],
    "use_unmatched": [True, False],
    "cosine_only": [False],
    "loss_weights": {
        "ce": [0.5, 1.0, 2.0],
        "contrastive": [0.05, 0.1, 0.3],
        "margin": [0.05, 0.1, 0.3],
        "consistency": [0.1, 0.2, 0.5],
        "entropy": [0.0005, 0.001, 0.005],
    },
}


def suggest_config(trial: optuna.Trial) -> dict:
    space = SEARCH_SPACE
    weights = {
        name: trial.suggest_categorical(f"loss_{name}", values)
        for name, values in space["loss_weights"].items()
    }
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", space["learning_rate"]["low"], space["learning_rate"]["high"], log=True
        ),
        "hidden": trial.suggest_categorical("hidden", space["hidden"]),
        "method": trial.suggest_categorical("method", space["method"]),
        "use_unmatched": trial.suggest_categorical("use_unmatched", space["use_unmatched"]),
        "cosine_only": trial.suggest_categorical("cosine_only", space["cosine_only"]),
        "loss_weights": weights,
    }


def trial_objective(samples, base_config, backbone, checkpoint_dir, split="validation"):
    """Build an Optuna objective that trains and scores one trial config."""
    from aurora_visual.evaluation.runner import evaluate
    from aurora_visual.training.engine import load_checkpoint

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        config = {**base_config, **suggest_config(trial)}
        path = checkpoint_dir / f"trial-{trial.number}.pt"
        meta, _ = train(samples, path, config, backbone)
        model, _, _ = load_checkpoint(path)
        metrics = evaluate(
            model, samples, split, meta["config"]["method"], confirmatory=False, interventions=False
        )
        value = float(metrics["macro_f1"])
        if value != value:  # NaN guard: fixture classes can be degenerate
            raise ValueError("Non-finite objective")
        trial.set_user_attr("data_kind", meta["data_kind"])
        trial.set_user_attr("checkpoint", str(path))
        return value

    return objective


def tune(samples, trials, base_config, backbone="local-color-v1", storage=None, seed=17):
    """Run TPE search; returns (study, best config, summary)."""
    if trials < 1:
        raise ValueError("trials must be positive")
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study_kwargs = {"sampler": sampler, "direction": "maximize"}
    if storage:
        Path(storage.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            load_if_exists=True, storage=storage, study_name="aurora-head", **study_kwargs
        )
    else:
        study = optuna.create_study(**study_kwargs)
    objective = trial_objective(samples, base_config, backbone, "artifacts/tuning/checkpoints")
    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    best_config = {
        "learning_rate": best.params["learning_rate"],
        "hidden": best.params["hidden"],
        "method": best.params["method"],
        "use_unmatched": best.params["use_unmatched"],
        "cosine_only": best.params["cosine_only"],
        "loss_weights": {name: best.params[f"loss_{name}"] for name in SEARCH_SPACE["loss_weights"]},
    }
    data_kind = best.user_attrs.get("data_kind", "fixture")
    summary = {
        "trials": len(study.trials),
        "best_value_validation_macro_f1": best.value,
        "best_trial_number": best.number,
        "best_config": best_config,
        "data_kind": data_kind,
        "tuning_split": "validation",
        "research_evaluated": False,
        "note": (
            "Tuning selects on the validation split only. Fixture tuning verifies the search "
            "wiring; it is not a research result and must not be reported as one."
            if data_kind == "fixture"
            else "Research tuning on validation only; test/calibration remain untouched."
        ),
        "trials_summary": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ],
        "data_hash": dataset_hash(samples),
    }
    return study, best_config, summary
