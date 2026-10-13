"""Configuration loading and the column naming scheme used throughout."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

N_FUEL_GROUPS = 4
N_SLOPE_BINS = 4


def embedding_columns(n_dims: int = 64) -> List[str]:
    """Column names of the mean pre-fire embedding vector."""
    return [f"emb_A{i:02d}" for i in range(n_dims)]


def fuel_slope_columns() -> List[str]:
    """Column names of the 16 fuel by slope conditional probability cells.

    Order is fuel-major: all four slope bins of fuel group 0, then fuel
    group 1, and so on.
    """
    return [
        f"p_high_f{f}_s{s}"
        for f in range(N_FUEL_GROUPS)
        for s in range(N_SLOPE_BINS)
    ]


def fuel_slope_codes() -> List[int]:
    """Integer codes used to encode fuel and slope in a single raster band."""
    return [f * 10 + s for f in range(N_FUEL_GROUPS) for s in range(N_SLOPE_BINS)]


def severity_histogram_columns() -> List[str]:
    """Column names of the per-fire severity class pixel counts."""
    return [f"sev_hist_{c}" for c in range(4)]


BASE_COLUMNS = [
    "fire_id",
    "fire_name",
    "event_date",
    "emb_year",
    "area_ha",
    "trn_elev",
    "trn_slope",
    "trn_aspect",
    "wth_temp_2m_mean",
    "wth_dew_2m_mean",
    "wth_precip_30d_sum",
]


@dataclass
class Config:
    """Typed view over ``config/default.yaml``."""

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: os.PathLike | str | None = None) -> "Config":
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as handle:
            return cls(raw=yaml.safe_load(handle))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    # Frequently used values, exposed as properties so call sites stay readable.
    @property
    def k_default(self) -> int:
        return int(self.raw["retrieval"]["k_default"])

    @property
    def k_list(self) -> List[int]:
        return list(self.raw["retrieval"]["k_list"])

    @property
    def prior_only(self) -> bool:
        return bool(self.raw["retrieval"]["prior_only"])

    @property
    def n_emb_dims(self) -> int:
        return int(self.raw["embedding"]["n_dims"])

    @property
    def severity_breaks(self) -> tuple:
        sev = self.raw["severity"]
        return (float(sev["t1"]), float(sev["t2"]), float(sev["t3"]))

    @property
    def high_class_min(self) -> int:
        return int(self.raw["severity"]["high_class_min"])

    @property
    def geo_norm_km(self) -> float:
        return float(self.raw["retrieval"]["geo_norm_km"])

    @property
    def doy_norm_days(self) -> float:
        return float(self.raw["retrieval"]["doy_norm_days"])

    @property
    def pixel_size_m(self) -> int:
        return int(self.raw["earth_engine"]["export_scale_m"])

    @property
    def pixel_area_ha(self) -> float:
        return (self.pixel_size_m ** 2) / 10_000.0
