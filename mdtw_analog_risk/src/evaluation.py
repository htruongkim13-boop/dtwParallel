from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class EventCluster:
    start: pd.Timestamp
    end: pd.Timestamp
    length: int


def compute_day_metrics(target: pd.Series, scores: pd.Series) -> Tuple[float, float]:
    mask = scores.notna() & target.notna()
    if mask.sum() < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(target[mask], scores[mask])),
        float(average_precision_score(target[mask], scores[mask])),
    )


def build_event_clusters(labels: pd.Series) -> pd.DataFrame:
    clusters = []
    in_cluster = False
    start = None
    for date, flag in labels.items():
        if flag and not in_cluster:
            in_cluster = True
            start = date
        if in_cluster and not flag:
            end = prev
            clusters.append(EventCluster(start=start, end=end, length=(end - start).days + 1))
            in_cluster = False
        prev = date
    if in_cluster:
        clusters.append(EventCluster(start=start, end=prev, length=(prev - start).days + 1))
    return pd.DataFrame([cluster.__dict__ for cluster in clusters])


def score_clusters(clusters: pd.DataFrame, scores: pd.Series) -> pd.Series:
    cluster_scores = []
    for _, row in clusters.iterrows():
        segment = scores.loc[row["start"]:row["end"]]
        cluster_scores.append(segment.max())
    return pd.Series(cluster_scores)


def build_noncritical_clusters(labels: pd.Series) -> pd.DataFrame:
    noncritical = (labels == 0).astype(int)
    return build_event_clusters(noncritical)


def compute_event_metrics(event_scores: pd.Series, non_event_scores: pd.Series) -> Tuple[float, float]:
    if event_scores.empty or non_event_scores.empty:
        return float("nan"), float("nan")
    y_true = np.concatenate([np.ones(len(event_scores)), np.zeros(len(non_event_scores))])
    y_scores = np.concatenate([event_scores.values, non_event_scores.values])
    return (
        float(roc_auc_score(y_true, y_scores)),
        float(average_precision_score(y_true, y_scores)),
    )
