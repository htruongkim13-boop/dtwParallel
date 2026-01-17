# MDTW Analog Risk Pipeline

该目录提供一个“多元DTW(13维) + 滑窗历史库(类比KPM) + 事前风险输出 + 日度评估(AUC/PR-AUC) + 事件级评估 + 图形输出(pdf+png)”的完整可运行示例。

## 1) 数据放置
请将以下文件放在 `mdtw_analog_risk/data/` 目录下：

- `result_with_original_prices.xlsx`
- `13个指标变量数据.csv`

## 2) 在仓库根目录应用 patch
本项目依赖对 `dtwParallel/dtw_functions.py` 的最小补丁（仅新增函数，不直接修改现有文件）。

在仓库根目录执行：

```bash
git apply mdtw_analog_risk/patches/dtwParallel_patch.diff
```

## 3) 运行流程
在仓库根目录执行：

```bash
python -m mdtw_analog_risk.src.run_pipeline
```

输出将写入：

```
mdtw_analog_risk/outputs/run_时间戳/
```

包含：
- `risk_scores.csv`
- `eval_day_level.csv`
- `eval_event_level.csv`
- `EUETS_Price_Over_Time.pdf/png`
- `Fig_risk_curve_w{w}_tau{tau}.pdf/png`
- `debug_non_null_rate.csv`
- `critical_clusters_for_eval.csv`
