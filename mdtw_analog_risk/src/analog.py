from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from joblib import Parallel, delayed

from .dtw_backend import dtw_distance


def prefilter_candidates(
    query_stats: np.ndarray,
    candidate_stats: np.ndarray,
    keep_pct: float,
    max_keep: int,
) -> np.ndarray:
    distances = np.linalg.norm(candidate_stats - query_stats, axis=1)
    total = len(distances)
    keep_n = min(max(1, int(total * keep_pct)), max_keep)
    keep_idx = np.argpartition(distances, keep_n - 1)[:keep_n]
    return keep_idx


def compute_topk_analogs(
    query: np.ndarray,
    candidates: np.ndarray,
    candidate_indices: np.ndarray,
    radius: int,
    topk: int,
    n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(candidate_indices) == 0:
        return np.array([]), np.array([])

    distances = Parallel(n_jobs=n_jobs)(
        delayed(dtw_distance)(query, candidates[idx], radius=radius, z_norm=True)
        for idx in candidate_indices
    )
    distances = np.array(distances, dtype=float)
    if len(distances) <= topk:
        order = np.argsort(distances)
    else:
        order = np.argpartition(distances, topk - 1)[:topk]
        order = order[np.argsort(distances[order])]
    return candidate_indices[order], distances[order]


def build_risk_score(
    analog_indices: np.ndarray,
    future_critical: np.ndarray,
) -> float:
    if len(analog_indices) == 0:
        return np.nan
    return float(future_critical[analog_indices].mean())


def summarize_prefilter(non_null_mask: np.ndarray) -> Dict[str, float]:
    return {
        "non_null_rate": float(non_null_mask.mean()),
        "non_null_count": int(non_null_mask.sum()),
        "total": int(len(non_null_mask)),
    }
