"""Fallback Gower distance implementation (numeric-only).

This lightweight module exists to avoid the external `gower` dependency when it cannot be
installed. It implements a minimal `gower_matrix` compatible with the usage in dtwParallel.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _to_numpy(data: "pd.DataFrame | np.ndarray | Iterable") -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        values = data.to_numpy(dtype=float)
    else:
        values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def gower_matrix(
    data: "pd.DataFrame | np.ndarray | Iterable", cat_features: Optional[Iterable[int]] = None
) -> np.ndarray:
    """Compute a Gower distance matrix for numeric-only data.

    Args:
        data: DataFrame or array-like of shape (n_samples, n_features).
        cat_features: Ignored in this numeric-only fallback.

    Returns:
        A (n_samples, n_samples) distance matrix with values in [0, 1].
    """
    values = _to_numpy(data)
    if values.size == 0:
        return np.zeros((0, 0))

    feature_min = np.nanmin(values, axis=0)
    feature_max = np.nanmax(values, axis=0)
    feature_range = feature_max - feature_min
    feature_range = np.where(feature_range == 0, 1.0, feature_range)

    diff = np.abs(values[:, None, :] - values[None, :, :])
    scaled = diff / feature_range

    # Use nanmean to ignore missing values.
    dist = np.nanmean(scaled, axis=2)
    dist = np.where(np.isnan(dist), 0.0, dist)
    return dist
