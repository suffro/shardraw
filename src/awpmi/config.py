from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    repository: str
    revision: str
    dtype: str
    device: str


@dataclass(frozen=True)
class NumericsConfig:
    accumulation_unit_roundoff: float


@dataclass(frozen=True)
class PagingConfig:
    page_widths: tuple[int, ...]


@dataclass(frozen=True)
class DatasetConfig:
    repository: str
    revision: str
    file: str
    text_column: str


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: DatasetConfig
    num_prompts: int
    min_prompt_tokens: int
    max_prompt_tokens: int
    seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    numerics: NumericsConfig
    paging: PagingConfig
    schedulers: tuple[str, ...]
    benchmark: BenchmarkConfig


def load_config(path: str | Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Return the typed config and the raw mapping (saved verbatim with each run)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    bench = raw["benchmark"]
    config = ExperimentConfig(
        model=ModelConfig(**raw["model"]),
        numerics=NumericsConfig(accumulation_unit_roundoff=float(raw["numerics"]["accumulation_unit_roundoff"])),
        paging=PagingConfig(page_widths=tuple(int(w) for w in raw["paging"]["page_widths"])),
        schedulers=tuple(raw["schedulers"]),
        benchmark=BenchmarkConfig(
            dataset=DatasetConfig(**bench["dataset"]),
            num_prompts=int(bench["num_prompts"]),
            min_prompt_tokens=int(bench["min_prompt_tokens"]),
            max_prompt_tokens=int(bench["max_prompt_tokens"]),
            seed=int(bench["seed"]),
        ),
    )
    return config, raw
