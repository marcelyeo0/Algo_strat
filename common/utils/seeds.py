"""Global seeding for numpy, Python's random module, torch and Optuna samplers."""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> dict[str, int]:
    """Seed every random number generator used in the project and return the seeds applied."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # torch is a hard dependency, but keep utilities importable without it
        pass
    return {"python": seed, "numpy": seed, "torch": seed, "optuna": seed}


def optuna_sampler(seed: int = DEFAULT_SEED):
    """Return a seeded TPE sampler so that tuning runs are reproducible."""
    import optuna

    return optuna.samplers.TPESampler(seed=seed)
