from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import yaml


@dataclass
class PrefilterConfig:
    keep_pct: float
    max_keep: int


@dataclass
class PipelineConfig:
    recent_n_rows: int
    w_list: List[int]
    tau_list: List[int]
    topk: int
    prefilter: PrefilterConfig
    dtw_radius: int
    only_noncritical_query: bool
    n_jobs: int


@dataclass
class PathsConfig:
    root: Path
    data_dir: Path
    output_root: Path


def load_config(config_path: Path) -> PipelineConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = yaml.safe_load(handle)

    prefilter_payload = payload.get("prefilter", {})
    prefilter = PrefilterConfig(
        keep_pct=float(prefilter_payload.get("keep_pct", 0.03)),
        max_keep=int(prefilter_payload.get("max_keep", 800)),
    )

    return PipelineConfig(
        recent_n_rows=int(payload.get("recent_n_rows", 600)),
        w_list=[int(item) for item in payload.get("w_list", [20])],
        tau_list=[int(item) for item in payload.get("tau_list", [20, 40, 60])],
        topk=int(payload.get("topk", 10)),
        prefilter=prefilter,
        dtw_radius=int(payload.get("dtw_radius", 10)),
        only_noncritical_query=bool(payload.get("only_noncritical_query", True)),
        n_jobs=int(payload.get("n_jobs", -1)),
    )


def build_paths_config(project_root: Path) -> PathsConfig:
    return PathsConfig(
        root=project_root,
        data_dir=project_root / "data",
        output_root=project_root / "outputs",
    )
