"""Bootstrap 95 % confidence intervals on accuracy and paired deltas."""
from __future__ import annotations

import numpy as np
from typing import Sequence


def accuracy_ci(per_sample_correct: Sequence[float], confidence: float = 0.95,
                 n_resamples: int = 10_000, seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, lo, hi) percentile bootstrap CI for binary correctness."""
    data = np.asarray(per_sample_correct, dtype=float)
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=float)
    n = len(data)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots[i] = data[idx].mean()
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1 - alpha)))
    return float(data.mean()), lo, hi


def paired_delta_ci(correct_A: Sequence[float], correct_B: Sequence[float],
                      confidence: float = 0.95,
                      n_resamples: int = 10_000, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap CI on accuracy_A - accuracy_B, item-matched.

    The two sequences MUST be the same length and aligned by sample.
    """
    a = np.asarray(correct_A, dtype=float)
    b = np.asarray(correct_B, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"sequence length mismatch: {len(a)} vs {len(b)}")
    rng = np.random.default_rng(seed)
    n = len(a)
    boots = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots[i] = a[idx].mean() - b[idx].mean()
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1 - alpha)))
    return float(a.mean() - b.mean()), lo, hi
