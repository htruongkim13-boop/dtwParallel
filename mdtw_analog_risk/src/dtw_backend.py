from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dtwParallel import dtw_functions  # noqa: E402


def dtw_distance(
    query: np.ndarray,
    candidate: np.ndarray,
    radius: Optional[int] = None,
    z_norm: bool = True,
) -> float:
    if hasattr(dtw_functions, "dtw_scalar"):
        return float(
            dtw_functions.dtw_scalar(
                query,
                candidate,
                sakoe_chiba_radius=radius,
                z_norm=z_norm,
            )
        )
    query_arr = np.asarray(query, dtype=float)
    candidate_arr = np.asarray(candidate, dtype=float)
    if z_norm:
        query_arr = _z_norm_2d(query_arr)
        candidate_arr = _z_norm_2d(candidate_arr)
    distance = dtw_functions.dtw(
        query_arr,
        candidate_arr,
        MTS=True,
        constrained_path_search="sakoe_chiba" if radius is not None else None,
        sakoe_chiba_radius=radius,
    )
    return float(distance)


def _z_norm_2d(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (data - mean) / std
