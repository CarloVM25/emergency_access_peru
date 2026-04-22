"""
utils.py
--------
Main pipeline runner for emergency_access_peru.

Executes the full data pipeline in order:
  1. Load all raw datasets        (data_loader.py)
  2. Clean all datasets           (cleaning.py)
  3. Geospatial joins & distances (geospatial.py)
  4. Index & metrics calculation  (metrics.py)

Run from the project root:
    python src/utils.py
"""

import sys
import os
import time

# Make sure sibling modules are importable when running as a script
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import (
    load_emergencias,
    load_ipress,
    load_ccpp,
    load_distritos,
)
from cleaning import (
    clean_emergencias,
    clean_ipress,
    clean_ccpp,
    clean_distritos,
)
from geospatial import (
    assign_facilities_to_districts,
    assign_ccpp_to_districts,
    compute_nearest_facility_distance,
    build_district_geodataframe,
)
from metrics import build_district_scores

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    """Print a clearly visible section header."""
    width = 65
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _elapsed(start: float) -> str:
    secs = time.time() - start
    return f"{secs:.1f}s" if secs < 60 else f"{secs/60:.1f}min"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_load() -> dict:
    """
    Run all load functions and return a dict of raw datasets.

    Returns
    -------
    dict with keys: 'emergencias', 'ipress', 'ccpp', 'distritos'
    """
    _section("STEP 1/4 — Load raw datasets")
    t = time.time()

    print("\n[1/4] load_emergencias()")
    emergencias = load_emergencias()

    print("\n[2/4] load_ipress()")
    ipress = load_ipress()

    print("\n[3/4] load_ccpp()")
    ccpp = load_ccpp()

    print("\n[4/4] load_distritos()")
    distritos = load_distritos()

    print(f"\n  Step complete in {_elapsed(t)}.")
    return {
        "emergencias": emergencias,
        "ipress":      ipress,
        "ccpp":        ccpp,
        "distritos":   distritos,
    }


def run_clean(raw: dict) -> dict:
    """
    Run all clean functions and return a dict of cleaned datasets.

    Parameters
    ----------
    raw : dict
        Output of run_load().

    Returns
    -------
    dict with keys: 'emergencias', 'ipress', 'ccpp', 'distritos'
    """
    _section("STEP 2/4 — Clean datasets")
    t = time.time()

    print("\n[1/4] clean_emergencias()")
    emergencias = clean_emergencias(raw["emergencias"])

    print("\n[2/4] clean_ipress()")
    ipress = clean_ipress(raw["ipress"])

    print("\n[3/4] clean_ccpp()")
    ccpp = clean_ccpp(raw["ccpp"])

    print("\n[4/4] clean_distritos()")
    distritos = clean_distritos(raw["distritos"])

    print(f"\n  Step complete in {_elapsed(t)}.")
    return {
        "emergencias": emergencias,
        "ipress":      ipress,
        "ccpp":        ccpp,
        "distritos":   distritos,
    }


def run_geospatial(cleaned: dict) -> dict:
    """
    Run all spatial joins and the nearest-facility distance calculation.

    Parameters
    ----------
    cleaned : dict
        Output of run_clean().

    Returns
    -------
    dict with keys:
        'ipress_districts'   -- IPRESS facilities with district attributes
        'ccpp_districts'     -- CCPP places with district attributes
        'ccpp_distances'     -- CCPP places with nearest-facility distance
        'district_summary'   -- district-level GeoDataFrame (all layers merged)
    """
    _section("STEP 3/4 — Geospatial joins & distances")
    t = time.time()

    print("\n[1/4] assign_facilities_to_districts()")
    ipress_districts = assign_facilities_to_districts(
        cleaned["ipress"], cleaned["distritos"]
    )

    print("\n[2/4] assign_ccpp_to_districts()")
    ccpp_districts = assign_ccpp_to_districts(
        cleaned["ccpp"], cleaned["distritos"]
    )

    print("\n[3/4] compute_nearest_facility_distance()")
    ccpp_distances = compute_nearest_facility_distance(
        cleaned["ccpp"], cleaned["ipress"]
    )

    print("\n[4/4] build_district_geodataframe()")
    district_summary = build_district_geodataframe(
        cleaned["distritos"],
        ipress_districts,
        ccpp_districts,
        ccpp_distances,
        cleaned["emergencias"],
    )

    print(f"\n  Step complete in {_elapsed(t)}.")
    return {
        "ipress_districts":  ipress_districts,
        "ccpp_districts":    ccpp_districts,
        "ccpp_distances":    ccpp_distances,
        "district_summary":  district_summary,
    }


def run_metrics(geo: dict) -> dict:
    """
    Compute baseline and alternative emergency access indices.

    Parameters
    ----------
    geo : dict
        Output of run_geospatial(). Uses 'district_summary'.

    Returns
    -------
    dict with key 'scores' -- district scores DataFrame (all indices,
    component scores, and categories for both specifications).
    """
    _section("STEP 4/4 — Metrics & index calculation")
    t = time.time()

    scores = build_district_scores(geo["district_summary"])

    print(f"\n  Step complete in {_elapsed(t)}.")
    return {"scores": scores}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> dict:
    """
    Execute the complete emergency_access_peru pipeline end-to-end.

    Steps
    -----
    1. Load     raw shapefiles and CSVs
    2. Clean    each dataset and write to data/processed/
    3. Geospatial  spatial joins + nearest-facility distances
    4. Metrics  baseline and alternative access indices

    Returns
    -------
    dict
        All pipeline outputs keyed by stage and dataset name.
    """
    pipeline_start = time.time()

    print("\n" + "#" * 65)
    print("#  emergency_access_peru — full pipeline")
    print("#" * 65)

    raw     = run_load()
    cleaned = run_clean(raw)
    geo     = run_geospatial(cleaned)
    metrics = run_metrics(geo)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    scores = metrics["scores"]
    dist   = geo["district_summary"]

    _section("PIPELINE COMPLETE")

    total = _elapsed(pipeline_start)

    print(f"\n  Total runtime : {total}")

    print("\n  --- Data volumes ---")
    print(f"  Raw emergencias records  : {len(raw['emergencias']):>8,}")
    print(f"  Clean emergencias records: {len(cleaned['emergencias']):>8,}")
    print(f"  IPRESS facilities (clean): {len(cleaned['ipress']):>8,}")
    print(f"  CCPP populated places    : {len(cleaned['ccpp']):>8,}")
    print(f"  Districts                : {len(dist):>8,}")

    print("\n  --- Geospatial results ---")
    n_fac_matched  = geo["ipress_districts"]["iddist"].notna().sum()
    n_ccpp_matched = geo["ccpp_districts"]["iddist"].notna().sum()
    avg_dist       = geo["ccpp_distances"]["dist_km"].mean()
    max_dist       = geo["ccpp_distances"]["dist_km"].max()
    print(f"  IPRESS facilities matched to a district : {n_fac_matched:,}")
    print(f"  CCPP places matched to a district       : {n_ccpp_matched:,}")
    print(f"  Mean distance to nearest emergency fac  : {avg_dist:.1f} km")
    print(f"  Max  distance to nearest emergency fac  : {max_dist:.1f} km")

    print("\n  --- Baseline index category breakdown ---")
    for cat in ["well_served", "above_average", "below_average", "underserved"]:
        n   = (scores["baseline_category"] == cat).sum()
        pct = n / len(scores) * 100
        print(f"  {cat:<16}: {n:>4} districts ({pct:.1f}%)")

    print("\n  --- Alternative index category breakdown ---")
    for cat in ["well_served", "above_average", "below_average", "underserved"]:
        n   = (scores["alt_category"] == cat).sum()
        pct = n / len(scores) * 100
        print(f"  {cat:<16}: {n:>4} districts ({pct:.1f}%)")

    n_changed = (scores["baseline_category"] != scores["alt_category"]).sum()
    print(f"\n  Districts changing category between specs: {n_changed} ({n_changed/len(scores)*100:.1f}%)")

    print("\n  --- Output files ---")
    outputs = [
        ("data/processed/", [
            "emergencias_clean.csv",
            "ipress_clean.csv",
            "ccpp_clean.shp",
            "distritos_clean.shp",
            "ipress_with_districts.gpkg",
            "ccpp_with_districts.gpkg",
            "ccpp_nearest_facility.gpkg",
            "districts_summary.gpkg",
            "districts_summary.shp",
        ]),
        ("output/tables/", [
            "district_scores.csv",
        ]),
    ]
    for folder, files in outputs:
        for f in files:
            print(f"  {folder}{f}")

    print()
    return {**cleaned, **geo, **metrics}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
