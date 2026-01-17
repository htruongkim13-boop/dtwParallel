from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(price: pd.Series) -> pd.Series:
    return price.pct_change().fillna(0.0)


def compute_delta(price: pd.Series) -> pd.Series:
    return price.diff().fillna(0.0)


def label_critical_days(returns: pd.Series, quantile: float = 0.05) -> pd.Series:
    threshold = returns.quantile(quantile)
    return (returns <= threshold).astype(int)


def compute_future_critical(critical: pd.Series, tau: int) -> pd.Series:
    values = critical.values
    future_flag = np.zeros_like(values, dtype=int)
    for idx in range(len(values)):
        end = min(len(values), idx + tau + 1)
        if idx + 1 < end:
            future_flag[idx] = int(values[idx + 1:end].max() > 0)
    return pd.Series(future_flag, index=critical.index, name=f"future_critical_{tau}")
