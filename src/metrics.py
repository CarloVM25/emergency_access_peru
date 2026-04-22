"""
metrics.py
----------
Builds two versions of a district-level emergency access index and
classifies districts into four policy-relevant categories.

Baseline specification  (weights: 40% facility, 30% activity, 30% distance)
    facility_score  = fac_count / ccpp_count           -- facilities per place
    activity_score  = tot_atend / fac_count            -- attendances per facility
    distance_score  = 1 / avg_dist_km                  -- inverse distance

Alternative specification  (weights: 25% facility, 25% activity, 50% distance)
    Same components but tot_atend is log-transformed before normalisation
    to reduce the influence of a few high-volume urban facilities.
    Distance weight is doubled to emphasise geographic access.

Classification  (both specs, quartile-based on each index)
    well_served     -- top 25 % (Q3 – max)
    above_average   -- Q2 – Q3
    below_average   -- Q1 – Q2
    underserved     -- bottom 25 % (min – Q1)
"""

import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "output", "tables")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEIGHTS_BASELINE    = {"facility": 0.40, "activity": 0.30, "distance": 0.30}
WEIGHTS_ALTERNATIVE = {"facility": 0.25, "activity": 0.25, "distance": 0.50}

# Ordered from worst to best — used for display and pd.Categorical
CATEGORY_ORDER = ["underserved", "below_average", "above_average", "well_served"]

# Columns to display in the top/bottom tables
_DISPLAY_COLS = [
    "distrito", "departamen",
    "fac_count", "tot_atend", "ccpp_count", "avg_dist_km",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _minmax(series: pd.Series) -> pd.Series:
    """
    Min-max normalise a Series to [0, 1].

    If all values are identical (range == 0), returns 0.5 uniformly so the
    resulting index is not undefined.
    """
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _classify(index_0_100: pd.Series) -> pd.Series:
    """
    Assign each district to one of four categories using quartile thresholds
    computed from the index itself.

    Quartile-based thresholds ensure that categories remain meaningful
    regardless of how skewed the index distribution is — which matters here
    because tot_atend is zero for ~66 % of districts.

    Returns a pd.Categorical with order: underserved < below_average <
    above_average < well_served.
    """
    q1 = index_0_100.quantile(0.25)
    q2 = index_0_100.quantile(0.50)
    q3 = index_0_100.quantile(0.75)

    def _label(v):
        if pd.isna(v):
            return "underserved"
        if v >= q3:
            return "well_served"
        if v >= q2:
            return "above_average"
        if v >= q1:
            return "below_average"
        return "underserved"

    labels = index_0_100.map(_label)
    return pd.Categorical(labels, categories=CATEGORY_ORDER, ordered=True)


def _raw_components(df: pd.DataFrame, log_activity: bool = False) -> pd.DataFrame:
    """
    Compute the three raw (pre-normalisation) component scores.

    Parameters
    ----------
    df            : district DataFrame with fac_count, tot_atend,
                    ccpp_count, avg_dist_km.
    log_activity  : if True, apply log1p to tot_atend before dividing by
                    fac_count (alternative spec).

    Returns
    -------
    pd.DataFrame with columns: facility_raw, activity_raw, distance_raw.
    """
    # --- facility score ---
    # facilities per populated place; clip ccpp_count at 1 to avoid /0
    # districts with ccpp_count=0 get 0 (no reference population → undefined)
    facility_raw = df["fac_count"] / df["ccpp_count"].clip(lower=1)
    facility_raw = facility_raw.where(df["ccpp_count"] > 0, other=0.0)

    # --- activity score ---
    # attendances per facility; districts with no facilities get 0
    atend = np.log1p(df["tot_atend"]) if log_activity else df["tot_atend"].astype(float)
    activity_raw = atend / df["fac_count"].clip(lower=1)
    activity_raw = activity_raw.where(df["fac_count"] > 0, other=0.0)

    # --- distance score ---
    # inverse distance: shorter distance → higher score
    # 3 districts have NaN avg_dist_km (no CCPP); assign 0 (worst access)
    dist_filled  = df["avg_dist_km"].fillna(df["avg_dist_km"].max())
    distance_raw = 1.0 / dist_filled.clip(lower=0.001)

    return pd.DataFrame({
        "facility_raw":  facility_raw,
        "activity_raw":  activity_raw,
        "distance_raw":  distance_raw,
    }, index=df.index)


def _build_index(
    df: pd.DataFrame,
    weights: dict,
    log_activity: bool = False,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Compute normalised component scores, weighted index (0–100), and category.

    Parameters
    ----------
    df           : district DataFrame.
    weights      : dict with keys 'facility', 'activity', 'distance'.
    log_activity : passed through to _raw_components.
    prefix       : column name prefix ('baseline_' or 'alt_').

    Returns
    -------
    pd.DataFrame with columns:
        {prefix}facility_score, {prefix}activity_score, {prefix}distance_score,
        {prefix}index, {prefix}category
    """
    raw = _raw_components(df, log_activity=log_activity)

    facility_score = _minmax(raw["facility_raw"])
    activity_score = _minmax(raw["activity_raw"])
    distance_score = _minmax(raw["distance_raw"])

    index_raw = (
        weights["facility"] * facility_score +
        weights["activity"] * activity_score +
        weights["distance"] * distance_score
    )
    index_0_100 = (index_raw * 100).round(2)

    return pd.DataFrame({
        f"{prefix}facility_score": facility_score.round(4),
        f"{prefix}activity_score": activity_score.round(4),
        f"{prefix}distance_score": distance_score.round(4),
        f"{prefix}index":          index_0_100,
        f"{prefix}category":       _classify(index_0_100),
    }, index=df.index)


def _print_table(title: str, df: pd.DataFrame, index_col: str, cat_col: str, n: int = 10) -> None:
    """Print a formatted top-N / bottom-N table."""
    cols = _DISPLAY_COLS + [index_col, cat_col]
    sorted_df = df[cols].sort_values(index_col, ascending=False)

    print(f"\n  {'TOP ' + str(n) + ' — ' + title}")
    print("  " + "-" * 90)
    print(sorted_df.head(n).to_string(index=False))

    print(f"\n  {'BOTTOM ' + str(n) + ' — ' + title}")
    print("  " + "-" * 90)
    print(sorted_df.tail(n).to_string(index=False))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_baseline_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the baseline emergency access index.

    Weights: 40% facility score, 30% activity score, 30% distance score.
    Activity score uses raw tot_atend (no log transformation).

    Parameters
    ----------
    df : district-level DataFrame with fac_count, tot_atend, ccpp_count,
         avg_dist_km columns.

    Returns
    -------
    pd.DataFrame
        Columns: baseline_facility_score, baseline_activity_score,
                 baseline_distance_score, baseline_index, baseline_category.
    """
    print("\n[compute_baseline_index]")
    print(f"  Weights: facility={WEIGHTS_BASELINE['facility']:.0%}  "
          f"activity={WEIGHTS_BASELINE['activity']:.0%}  "
          f"distance={WEIGHTS_BASELINE['distance']:.0%}")
    print("  Activity: raw tot_atend / fac_count")

    result = _build_index(df, WEIGHTS_BASELINE, log_activity=False, prefix="baseline_")

    cats = result["baseline_category"].value_counts().reindex(CATEGORY_ORDER)
    for cat, n in cats.items():
        print(f"  {cat:<16}: {n:>4} districts")

    return result


def compute_alternative_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the alternative emergency access index.

    Weights: 25% facility score, 25% activity score, 50% distance score.
    Activity score uses log1p(tot_atend) / fac_count to dampen the effect
    of high-volume urban facilities on the overall ranking.

    Parameters
    ----------
    df : district-level DataFrame (same as compute_baseline_index).

    Returns
    -------
    pd.DataFrame
        Columns: alt_facility_score, alt_activity_score,
                 alt_distance_score, alt_index, alt_category.
    """
    print("\n[compute_alternative_index]")
    print(f"  Weights: facility={WEIGHTS_ALTERNATIVE['facility']:.0%}  "
          f"activity={WEIGHTS_ALTERNATIVE['activity']:.0%}  "
          f"distance={WEIGHTS_ALTERNATIVE['distance']:.0%}")
    print("  Activity: log1p(tot_atend) / fac_count")

    result = _build_index(df, WEIGHTS_ALTERNATIVE, log_activity=True, prefix="alt_")

    cats = result["alt_category"].value_counts().reindex(CATEGORY_ORDER)
    for cat, n in cats.items():
        print(f"  {cat:<16}: {n:>4} districts")

    return result


def compare_specifications(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Compare category assignments between baseline and alternative specs.

    Prints the number of districts that change category and a cross-tabulation
    showing which categories districts move between.

    Parameters
    ----------
    scores : DataFrame containing 'baseline_category' and 'alt_category'.

    Returns
    -------
    pd.DataFrame
        Cross-tabulation (baseline rows × alternative columns).
    """
    print("\n[compare_specifications]")

    n_changed = (scores["baseline_category"] != scores["alt_category"]).sum()
    n_total   = len(scores)
    pct       = n_changed / n_total * 100

    print(f"  Districts that change category: {n_changed:,} / {n_total:,} ({pct:.1f}%)")

    # Direction of movement for changed districts
    changed = scores[scores["baseline_category"] != scores["alt_category"]].copy()
    changed["direction"] = changed.apply(
        lambda r: "improved" if r["alt_category"] > r["baseline_category"] else "worsened",
        axis=1,
    )
    improved = (changed["direction"] == "improved").sum()
    worsened = (changed["direction"] == "worsened").sum()
    print(f"  Improved in alternative spec  : {improved:,} districts")
    print(f"  Worsened in alternative spec  : {worsened:,} districts")

    # Cross-tabulation
    crosstab = pd.crosstab(
        scores["baseline_category"],
        scores["alt_category"],
        rownames=["baseline"],
        colnames=["alternative"],
    )
    # Reindex so both axes follow the correct category order
    crosstab = crosstab.reindex(index=CATEGORY_ORDER, columns=CATEGORY_ORDER, fill_value=0)

    print("\n  Category cross-tabulation (rows=baseline, cols=alternative):")
    print("  " + crosstab.to_string().replace("\n", "\n  "))

    return crosstab


def build_district_scores(district_gdf: gpd.GeoDataFrame = None) -> pd.DataFrame:
    """
    Orchestrate the full metrics pipeline:
      1. Load district summary (if not provided).
      2. Compute baseline and alternative indices.
      3. Print top-10 and bottom-10 rankings for each spec.
      4. Compare category assignments across specs.
      5. Save results to output/tables/district_scores.csv.

    Parameters
    ----------
    district_gdf : optional GeoDataFrame from build_district_geodataframe().
                   If None, loaded from data/processed/districts_summary.gpkg.

    Returns
    -------
    pd.DataFrame
        All raw values, component scores, indices, and categories per district.
    """
    print("\n[build_district_scores]")

    # --- Load data ---
    if district_gdf is None:
        path = os.path.join(PROCESSED_DIR, "districts_summary.gpkg")
        print(f"  Loading district summary from {path} ...")
        district_gdf = gpd.read_file(path)

    # Work on a plain DataFrame — geometry not needed for metrics
    df = pd.DataFrame(district_gdf.drop(columns="geometry", errors="ignore"))
    print(f"  Districts loaded: {len(df):,}")

    # --- Compute indices ---
    baseline = compute_baseline_index(df)
    alt      = compute_alternative_index(df)

    # --- Assemble full results table ---
    id_cols = ["iddist", "distrito", "departamen", "provincia",
               "fac_count", "tot_atend", "ccpp_count", "avg_dist_km"]
    scores = pd.concat([df[id_cols], baseline, alt], axis=1)

    # --- Print rankings ---
    print("\n" + "=" * 92)
    print("  BASELINE SPECIFICATION  (40% facility | 30% activity | 30% distance)")
    print("=" * 92)
    _print_table("BASELINE", scores, "baseline_index", "baseline_category")

    print("\n" + "=" * 92)
    print("  ALTERNATIVE SPECIFICATION  (25% facility | 25% activity | 50% distance | log activity)")
    print("=" * 92)
    _print_table("ALTERNATIVE", scores, "alt_index", "alt_category")

    # --- Compare specs ---
    print("\n" + "=" * 92)
    print("  SPECIFICATION COMPARISON")
    print("=" * 92)
    compare_specifications(scores)

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "district_scores.csv")
    scores.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n  Saved -> {out_path}  ({len(scores):,} rows x {len(scores.columns)} cols)")

    return scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scores = build_district_scores()
