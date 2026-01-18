from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class WindowSpec:
    start: int
    end: int


@dataclass
class WindowedDataset:
    windows: List[WindowSpec]
    matrix: np.ndarray
    index: pd.Index


def build_windows(data: pd.DataFrame, window: int) -> WindowedDataset:
    values = data.values
    windows: List[WindowSpec] = []
    slices = []
    for end in range(window - 1, len(values)):
        start = end - window + 1
        windows.append(WindowSpec(start=start, end=end))
        slices.append(values[start:end + 1])
    matrix = np.stack(slices, axis=0) if slices else np.empty((0, window, values.shape[1]))
    index = data.index[window - 1:]
    return WindowedDataset(windows=windows, matrix=matrix, index=index)


def compute_window_stats(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1)
    return np.concatenate([means, stds], axis=1)
