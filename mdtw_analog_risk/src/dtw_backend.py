from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def _ensure_gower_dependency() -> None:
    try:
        importlib.import_module("gower")
        return
    except ImportError:
        pass

    fallback_path = REPO_ROOT / "0118" / "gower.py"
    if not fallback_path.exists():
        return

    spec = importlib.util.spec_from_file_location("gower", fallback_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["gower"] = module


_ensure_gower_dependency()


def _load_dtw_functions():
    spec = importlib.util.find_spec("dtwParallel.dtw_functions")
    if spec is not None:
        return importlib.import_module("dtwParallel.dtw_functions")

    dtw_path = REPO_ROOT / "dtwParallel" / "dtw_functions.py"
    if dtw_path.exists():
        module_name = "dtwParallel_dtw_functions_fallback"
        spec = importlib.util.spec_from_file_location(module_name, dtw_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    raise ImportError("Unable to locate dtwParallel.dtw_functions")


dtw_functions = _load_dtw_functions()


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
