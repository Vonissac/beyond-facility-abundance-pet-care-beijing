from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "reproducibility" / "output"
TABLE_OUT = OUT / "tables"
FIG_OUT = OUT / "figures"

RANDOM_STATE = 20260609
BASE_DRAWS = 5000
SPATIAL_DRAWS = 2000
MISCLASS_SIMULATIONS = 1000

PAIR_SPECS = [
    ("pet_retail__pet_medical", "pet_retail_buffer_1p0km_pct", "pet_medical_buffer_1p0km_pct", "companion_animal_pair"),
    ("pet_retail__elder_care", "pet_retail_buffer_1p0km_pct", "elder_care_buffer_1p0km_pct", "pet_formal_care_contrast"),
    ("pet_medical__elder_care", "pet_medical_buffer_1p0km_pct", "elder_care_buffer_1p0km_pct", "pet_formal_care_contrast"),
    ("elder_care__health_care", "elder_care_buffer_1p0km_pct", "health_care_buffer_1p0km_pct", "formal_care_pair"),
    ("health_care__transit", "health_care_buffer_1p0km_pct", "transit_buffer_1p0km_pct", "urban_service_pair"),
    ("daily_life__health_care", "daily_life_buffer_1p0km_pct", "health_care_buffer_1p0km_pct", "urban_service_pair"),
    ("education_child__recreation", "education_child_buffer_1p0km_pct", "recreation_park_sport_buffer_1p0km_pct", "age_related_public_life_pair"),
    ("daily_life__transit", "daily_life_buffer_1p0km_pct", "transit_buffer_1p0km_pct", "urban_service_pair"),
]

CONTEXTS = {
    "all_observed": lambda d: d["population_observed"].astype(bool),
    "non_elite": lambda d: d["flag_non_elite"].astype(bool),
    "daily_life_top_non_elite": lambda d: d["flag_daily_life_top"].astype(bool) & d["flag_non_elite"].astype(bool),
    "population_dense": lambda d: d["flag_pop_dense"].astype(bool),
}

STRATA_SPECS = {
    "base_context_q4": ["population_pct", "housing_price_pct", "daily_life_buffer_1p0km_pct", "service_density_pct"],
    "no_service_density_q4": ["population_pct", "housing_price_pct", "daily_life_buffer_1p0km_pct"],
    "socio_spatial_q4": ["population_pct", "housing_price_pct", "lon_center", "lat_center"],
    "base_plus_spatial_q4": [
        "population_pct",
        "housing_price_pct",
        "daily_life_buffer_1p0km_pct",
        "service_density_pct",
        "lon_center",
        "lat_center",
    ],
    "fine_base_context_q5": ["population_pct", "housing_price_pct", "daily_life_buffer_1p0km_pct", "service_density_pct"],
}


def boolish(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_grid() -> pd.DataFrame:
    df = pd.read_csv(DATA / "grid" / "grid_1km.csv")
    for col in ["population_observed", "flag_non_elite", "flag_daily_life_top", "flag_pop_dense"]:
        df[col] = boolish(df[col])
    return df


def qcut_labels(df: pd.DataFrame, cols: list[str], q: int) -> pd.Series:
    labels: list[pd.Series] = []
    for col in cols:
        values = df[col].copy()
        values = values.fillna(-1 if col == "housing_price_pct" else values.median())
        try:
            label = pd.qcut(values.rank(method="first"), q, labels=False, duplicates="drop").astype(int).astype(str)
        except ValueError:
            label = pd.Series(["0"] * len(df), index=df.index)
        labels.append(label)
    out = labels[0]
    for label in labels[1:]:
        out = out + "_" + label
    return out


def expected_perm(df: pd.DataFrame, strata: pd.Series, a: pd.Series, b: pd.Series, n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(strata.to_numpy(), sort=False)
    a_arr = np.asarray(a, dtype=np.int16)
    b_arr = np.asarray(b, dtype=np.int16)
    observed = int((a_arr * b_arr).sum())
    expected = 0.0
    variance = 0.0
    groups: list[tuple[int, int, int]] = []
    counts = np.bincount(codes)
    a_counts = np.bincount(codes, weights=a_arr).astype(int)
    b_counts = np.bincount(codes, weights=b_arr).astype(int)
    for n, a_n, b_n in zip(counts, a_counts, b_counts):
        if n < 4:
            continue
        pa, pb = a_n / n, b_n / n
        p = pa * pb
        expected += n * p
        variance += n * p * max(1 - p, 0)
        groups.append((n, a_n, b_n))
    sims = np.zeros(n_perm, dtype=np.float32)
    for n, a_n, b_n in groups:
        sims += rng.hypergeometric(ngood=b_n, nbad=n - b_n, nsample=a_n, size=n_perm).astype(np.float32)
    q025, q50, q975 = np.quantile(sims, [0.025, 0.5, 0.975])
    return {
        "O": observed,
        "E": expected,
        "Delta": observed - expected,
        "rho": observed / expected if expected else np.nan,
        "z_approx": (observed - expected) / math.sqrt(variance) if variance > 0 else np.nan,
        "q2.5": q025,
        "q50": q50,
        "q97.5": q975,
        "M": observed - q975,
        "p_upper": (float((sims >= observed).sum()) + 1.0) / (n_perm + 1.0),
        "R": n_perm,
        "seed": seed,
        "usable_grids": int(sum(n for n, _, _ in groups)),
        "strata_used": len(groups),
    }


def ensure_output_dirs() -> None:
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
