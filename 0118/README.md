# 0118 Notebook Runner

本目录用于在 **Jupyter Notebook** 中运行 `mdtw_analog_risk`，并提供一个
**本地 Gower 距离实现**（`gower.py`），解决外部 `gower` 依赖无法下载的问题。

## 为什么需要 `gower.py`
`dtwParallel` 默认依赖第三方 `gower` 包。如果你无法安装该包，可以在 Notebook 中
将本目录加入 `sys.path`，这样 `import gower` 会优先加载这里的 `gower.py`。

## Notebook 运行步骤（推荐）
1. 启动 Jupyter Notebook，并打开仓库根目录或 `0118/` 目录。
2. 在 Notebook 中执行以下代码块，确保路径与 Gower 回退生效：

```python
from pathlib import Path
import sys

root = Path.cwd()
while root != root.parent and not (root / "mdtw_analog_risk").exists():
    root = root.parent

sys.path.insert(0, str(root / "0118"))  # 优先加载 0118/gower.py
sys.path.insert(0, str(root))
```

3. （可选）快速试跑：将 `recent_n_rows` 改为 200：

```python
import yaml

config_path = root / "mdtw_analog_risk" / "config.yaml"
with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["recent_n_rows"] = 200

with config_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)
```

4. 运行主流程：

```python
from mdtw_analog_risk.src.run_pipeline import main

main()
```

## 分布式/并行运行提示
`mdtw_analog_risk` 内部使用 `joblib` 并行计算。如果需要最大化并行度，确保
`mdtw_analog_risk/config.yaml` 中 `n_jobs: -1`。
