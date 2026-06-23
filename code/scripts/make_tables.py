from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    BASE_DRAWS,
    CONTEXTS,
    MISCLASS_SIMULATIONS,
    PAIR_SPECS,
    RANDOM_STATE,
    SPATIAL_DRAWS,
    STRATA_SPECS,
    TABLE_OUT,
    ensure_output_dirs,
    expected_perm,
    load_grid,
    qcut_labels,
)


def run_base_ccem(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    seed_rows = []
    for context, mask_func in CONTEXTS.items():
        ctx = df[mask_func(df)].copy()
        strata = qcut_labels(ctx, STRATA_SPECS["base_context_q4"], 4)
        for threshold in [0.70, 0.80, 0.90]:
            for pair_name, col_a, col_b, family in PAIR_SPECS:
                stats = expected_perm(ctx, strata, ctx[col_a].ge(threshold), ctx[col_b].ge(threshold), BASE_DRAWS, RANDOM_STATE)
                rows.append(
                    {
                        "component": "base_ccem",
                        "context": context,
                        "strata_spec": "base_context_q4",
                        "threshold": threshold,
                        "pair": pair_name,
                        "pair_family": family,
                        "grids": len(ctx),
                        **stats,
                    }
                )
    for seed in [20260609, 20260610, 20260611, 20260612, 20260613]:
        ctx = df[CONTEXTS["all_observed"](df)].copy()
        strata = qcut_labels(ctx, STRATA_SPECS["base_context_q4"], 4)
        stats = expected_perm(
            ctx,
            strata,
            ctx["pet_retail_buffer_1p0km_pct"].ge(0.80),
            ctx["pet_medical_buffer_1p0km_pct"].ge(0.80),
            1000,
            seed,
        )
        seed_rows.append(
            {
                "component": "base_ccem_seed_sensitivity",
                "context": "all_observed",
                "threshold": 0.80,
                "pair": "pet_retail__pet_medical",
                **stats,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(seed_rows)


def run_spatial_ccem(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    seed_rows = []
    for context, mask_func in CONTEXTS.items():
        ctx = df[mask_func(df)].copy()
        for spec, cols in STRATA_SPECS.items():
            strata = qcut_labels(ctx, cols, 5 if spec == "fine_base_context_q5" else 4)
            for threshold in [0.80, 0.90]:
                for pair_name, col_a, col_b, family in PAIR_SPECS[:6]:
                    stats = expected_perm(ctx, strata, ctx[col_a].ge(threshold), ctx[col_b].ge(threshold), SPATIAL_DRAWS, RANDOM_STATE)
                    rows.append(
                        {
                            "component": "spatial_ccem",
                            "context": context,
                            "strata_spec": spec,
                            "threshold": threshold,
                            "pair": pair_name,
                            "pair_family": family,
                            "grids": len(ctx),
                            **stats,
                        }
                    )
    for seed in [20260609, 20260610, 20260611, 20260612, 20260613]:
        ctx = df[CONTEXTS["all_observed"](df)].copy()
        strata = qcut_labels(ctx, STRATA_SPECS["base_plus_spatial_q4"], 4)
        stats = expected_perm(
            ctx,
            strata,
            ctx["pet_retail_buffer_1p0km_pct"].ge(0.80),
            ctx["pet_medical_buffer_1p0km_pct"].ge(0.80),
            1000,
            seed,
        )
        seed_rows.append(
            {
                "component": "spatial_ccem_seed_sensitivity",
                "context": "all_observed",
                "strata_spec": "base_plus_spatial_q4",
                "threshold": 0.80,
                "pair": "pet_retail__pet_medical",
                **stats,
            }
        )
    tests = pd.DataFrame(rows)
    pet = tests[tests["pair"].eq("pet_retail__pet_medical")]
    summary = (
        pet.groupby("context")
        .agg(
            tests=("pair", "count"),
            above_q97_5=("M", lambda x: int((x > 0).sum())),
            min_rho=("rho", "min"),
            median_rho=("rho", "median"),
            min_M=("M", "min"),
        )
        .reset_index()
    )
    return tests, summary, pd.DataFrame(seed_rows)


def strata_transparency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ctx = df[CONTEXTS["all_observed"](df)].copy()
    for spec, cols in STRATA_SPECS.items():
        q = 5 if spec == "fine_base_context_q5" else 4
        strata = qcut_labels(ctx, cols, q)
        sizes = strata.value_counts()
        rows.append(
            {
                "specification": spec,
                "matching_variables": "; ".join(cols),
                "binning": f"rank-based q{q}",
                "usable_grids": len(ctx),
                "strata": len(sizes),
                "min_size": int(sizes.min()),
                "median_size": float(sizes.median()),
                "max_size": int(sizes.max()),
                "singleton_strata": int((sizes == 1).sum()),
                "small_strata_handling": "strata with n < 4 are excluded from permutation shuffling",
            }
        )
    return pd.DataFrame(rows)


def pet_retail_misclassification(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    ctx = df[CONTEXTS["all_observed"](df)].copy().reset_index(drop=True)
    strata = qcut_labels(ctx, STRATA_SPECS["base_context_q4"], 4)
    med_high = ctx["pet_medical_buffer_1p0km_pct"].ge(0.80)
    retail_counts = ctx["pet_retail_within_1p0km"].fillna(0).astype(int).to_numpy()
    med_high_arr = med_high.to_numpy()
    scenarios = [
        ("baseline", 1.0, 0.0),
        ("10pct_removal", 0.90, 0.10),
        ("15p8pct_removal", 0.842, 0.158),
        ("20pct_removal", 0.80, 0.20),
    ]
    rows = []
    for name, retained, remove in scenarios:
        rhos = []
        margins = []
        pass_count = 0
        sims = 1 if remove == 0 else MISCLASS_SIMULATIONS
        for i in range(sims):
            retained_counts = retail_counts if remove == 0 else rng.binomial(retail_counts, retained)
            pct = pd.Series(retained_counts).rank(method="min") - 1
            denom = max(len(pct) - 1, 1)
            retail_high = (pct / denom).ge(0.80)
            stats = expected_perm(ctx, strata, retail_high, pd.Series(med_high_arr), 1000 if remove == 0 else 300, RANDOM_STATE + i)
            rhos.append(stats["rho"])
            margins.append(stats["M"])
            pass_count += int(stats["M"] > 0)
        rows.append(
            {
                "scenario": name,
                "retail_poi_retained_share": retained,
                "simulations": sims if remove else 0,
                "median_rho": float(np.median(rhos)),
                "rho_q2.5": float(np.quantile(rhos, 0.025)),
                "rho_q97.5": float(np.quantile(rhos, 0.975)),
                "median_M": float(np.median(margins)),
                "share_O_above_q97.5": pass_count / sims,
                "interpretation": "Stable" if pass_count / sims >= 0.95 else ("Weakened but mostly positive" if pass_count / sims >= 0.5 else "Fragile"),
            }
        )
    return pd.DataFrame(rows)


def component_summary(base: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    pet_base = base[(base["pair"].eq("pet_retail__pet_medical")) & (base["threshold"].eq(0.80))]
    pet_spatial = spatial[(spatial["pair"].eq("pet_retail__pet_medical")) & (spatial["threshold"].eq(0.80))]
    return pd.DataFrame(
        [
            {
                "component": "Base CCEM",
                "permutation_draws": BASE_DRAWS,
                "seed": RANDOM_STATE,
                "pet_rows_above_q97.5": f"{int((pet_base['M'] > 0).sum())}/{len(pet_base)}",
                "min_pet_rho": pet_base["rho"].min(),
            },
            {
                "component": "Spatial CCEM",
                "permutation_draws": SPATIAL_DRAWS,
                "seed": RANDOM_STATE,
                "pet_rows_above_q97.5": f"{int((pet_spatial['M'] > 0).sum())}/{len(pet_spatial)}",
                "min_pet_rho": pet_spatial["rho"].min(),
            },
        ]
    )


def main() -> None:
    ensure_output_dirs()
    df = load_grid()
    base, base_seed = run_base_ccem(df)
    spatial, spatial_summary, spatial_seed = run_spatial_ccem(df)
    tables = {
        "base_ccem": base,
        "base_seed": base_seed,
        "spatial_ccem": spatial,
        "spatial_summary": spatial_summary,
        "spatial_seed": spatial_seed,
        "strata": strata_transparency(df),
        "misclass": pet_retail_misclassification(df),
        "components": component_summary(base, spatial),
    }
    for name, table in tables.items():
        table.to_csv(TABLE_OUT / f"{name}.csv", index=False)
    print("Core tables reproduced.")


if __name__ == "__main__":
    main()
