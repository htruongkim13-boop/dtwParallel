from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .analog import build_risk_score, compute_topk_analogs, prefilter_candidates, summarize_prefilter
from .config import build_paths_config, load_config
from .data_io import align_data, read_factor_data, read_price_data, select_recent_rows, validate_factor_columns
from .evaluation import (
    build_event_clusters,
    build_noncritical_clusters,
    compute_day_metrics,
    compute_event_metrics,
    score_clusters,
)
from .feature_engineering import compute_delta, compute_future_critical, compute_returns, label_critical_days
from .plotting import plot_price, plot_risk_curve
from .windowing import build_windows, compute_window_stats


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = build_paths_config(project_root)
    config = load_config(project_root / "config.yaml")

    price = read_price_data(paths.data_dir / "result_with_original_prices.xlsx")
    factors = read_factor_data(paths.data_dir / "13个指标变量数据.csv")
    price, factors = align_data(price, factors)
    price, factors = select_recent_rows(price, factors, config.recent_n_rows)

    factor_cols = validate_factor_columns(factors)
    factors = factors[factor_cols]

    returns = compute_returns(price["price"])
    delta = compute_delta(price["price"])
    critical = label_critical_days(returns)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = paths.output_root / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    price_output = price.copy()
    price_output["delta"] = delta
    price_output["returns"] = returns
    price_output["critical"] = critical

    plot_price(price["price"], critical, output_dir)

    risk_records = []
    eval_day_records = []
    eval_event_records = []
    debug_records = []

    for w in config.w_list:
        windowed = build_windows(factors, w)
        if windowed.matrix.size == 0:
            continue
        window_stats = compute_window_stats(windowed.matrix)
        critical_window = critical.loc[windowed.index]

        for tau in config.tau_list:
            future_critical = compute_future_critical(critical, tau).loc[windowed.index]
            risk_scores = pd.Series(index=windowed.index, dtype=float)
            non_null_mask = np.zeros(len(windowed.index), dtype=bool)

            for idx in range(len(windowed.index)):
                if config.only_noncritical_query and critical_window.iloc[idx] == 1:
                    risk_scores.iloc[idx] = np.nan
                    continue
                candidate_end = idx - tau
                if candidate_end <= 0:
                    risk_scores.iloc[idx] = np.nan
                    continue
                candidate_indices = np.arange(0, candidate_end)
                if candidate_indices.size == 0:
                    risk_scores.iloc[idx] = np.nan
                    continue

                keep_idx = prefilter_candidates(
                    window_stats[idx],
                    window_stats[candidate_indices],
                    config.prefilter.keep_pct,
                    config.prefilter.max_keep,
                )
                filtered_indices = candidate_indices[keep_idx]

                top_idx, _ = compute_topk_analogs(
                    windowed.matrix[idx],
                    windowed.matrix,
                    filtered_indices,
                    radius=config.dtw_radius,
                    topk=config.topk,
                    n_jobs=config.n_jobs,
                )
                risk_scores.iloc[idx] = build_risk_score(top_idx, future_critical.values)
                non_null_mask[idx] = not np.isnan(risk_scores.iloc[idx])

            for date, score in risk_scores.items():
                risk_records.append(
                    {
                        "date": date,
                        "w": w,
                        "tau": tau,
                        "risk_score": score,
                        "future_critical": future_critical.loc[date],
                    }
                )

            auc, pr_auc = compute_day_metrics(future_critical, risk_scores)
            eval_day_records.append(
                {
                    "w": w,
                    "tau": tau,
                    "auc": auc,
                    "pr_auc": pr_auc,
                    "count": int(future_critical.notna().sum()),
                    "non_null_risk": int(risk_scores.notna().sum()),
                }
            )

            clusters = build_event_clusters(critical_window)
            noncritical_clusters = build_noncritical_clusters(critical_window)
            event_scores = score_clusters(clusters, risk_scores) if not clusters.empty else pd.Series(dtype=float)
            non_event_scores = (
                score_clusters(noncritical_clusters, risk_scores)
                if not noncritical_clusters.empty
                else pd.Series(dtype=float)
            )
            event_auc, event_pr_auc = compute_event_metrics(event_scores, non_event_scores)
            eval_event_records.append(
                {
                    "w": w,
                    "tau": tau,
                    "event_auc": event_auc,
                    "event_pr_auc": event_pr_auc,
                    "event_count": int(len(event_scores)),
                    "non_event_count": int(len(non_event_scores)),
                }
            )

            debug_info = summarize_prefilter(non_null_mask)
            debug_info.update({"w": w, "tau": tau})
            debug_records.append(debug_info)

            if not clusters.empty:
                clusters.to_csv(output_dir / "critical_clusters_for_eval.csv", index=False)

            plot_risk_curve(risk_scores, future_critical, output_dir, w, tau)

    risk_df = pd.DataFrame(risk_records)
    risk_df.to_csv(output_dir / "risk_scores.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    pd.DataFrame(eval_day_records).to_csv(output_dir / "eval_day_level.csv", index=False)
    pd.DataFrame(eval_event_records).to_csv(output_dir / "eval_event_level.csv", index=False)
    pd.DataFrame(debug_records).to_csv(output_dir / "debug_non_null_rate.csv", index=False)


if __name__ == "__main__":
    main()
