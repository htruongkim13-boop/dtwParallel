from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import pandas as pd


def _infer_date_column(frame: pd.DataFrame) -> str:
    candidates = [col for col in frame.columns if "date" in col.lower()]
    if candidates:
        return candidates[0]
    return frame.columns[0]


def _infer_price_column(frame: pd.DataFrame) -> str:
    candidates = [col for col in frame.columns if "price" in col.lower()]
    if candidates:
        return candidates[0]
    numeric_cols = frame.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric column found for price data.")
    return numeric_cols[0]


def read_price_data(price_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(price_path)
    date_col = _infer_date_column(frame)
    price_col = _infer_price_column(frame)
    frame = frame[[date_col, price_col]].rename(columns={date_col: "date", price_col: "price"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date")
    frame = frame.set_index("date")
    return frame


def read_factor_data(factor_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(factor_path)
    date_col = _infer_date_column(frame)
    frame = frame.rename(columns={date_col: "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date")
    frame = frame.set_index("date")
    return frame


def align_data(price: pd.DataFrame, factors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common_index = price.index.intersection(factors.index)
    price_aligned = price.loc[common_index].copy()
    factors_aligned = factors.loc[common_index].copy()
    return price_aligned, factors_aligned


def select_recent_rows(price: pd.DataFrame, factors: pd.DataFrame, n_rows: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if n_rows <= 0:
        return price, factors
    price_recent = price.tail(n_rows)
    factors_recent = factors.loc[price_recent.index]
    return price_recent, factors_recent


def validate_factor_columns(factors: pd.DataFrame) -> List[str]:
    numeric_cols = factors.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("Factor data has no numeric columns.")
    return numeric_cols
