"""
geospatial.py
-------------
Spatial analysis functions for the emergency_access_peru pipeline.

All distance calculations reproject to EPSG:32718 (WGS 84 / UTM Zone 18S),
the standard projected CRS for Peru, where coordinate units are metres.

Public functions
----------------
assign_facilities_to_districts  -- spatial join: IPRESS → districts
assign_ccpp_to_districts         -- spatial join: populated places → districts
compute_nearest_facility_distance -- nearest emergency facility per CCPP point
build_district_geodataframe      -- aggregate all layers into district summary
"""

import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Projected CRS for Peru — UTM Zone 18S, units: metres
_CRS_GEO  = "EPSG:4326"
_CRS_PROJ = "EPSG:32718"

# IPRESS categories and facility type that indicate emergency-capable services
_EMERGENCY_CATEGORIAS = {"II-1", "II-2", "III-1", "III-2", "II-E", "III-E"}
_EMERGENCY_TIPO       = "ESTABLECIMIENTO DE SALUD CON INTERNAMIENTO"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _processed(filename: str) -> str:
    return os.path.join(PROCESSED_DIR, filename)


def _ipress_to_geodataframe(ipress_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert the IPRESS flat DataFrame to a GeoDataFrame using lon/lat."""
    gdf = gpd.GeoDataFrame(
        ipress_df,
        geometry=gpd.points_from_xy(ipress_df["lon"], ipress_df["lat"]),
        crs=_CRS_GEO,
    )
    return gdf


def _filter_emergency_facilities(ipress_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows whose facility type or category indicates emergency services.

    Criteria (union):
      - tipo == 'ESTABLECIMIENTO DE SALUD CON INTERNAMIENTO'
      - categoria in {II-1, II-2, III-1, III-2, II-E, III-E}
    """
    mask = (
        ipress_df["tipo"].eq(_EMERGENCY_TIPO) |
        ipress_df["categoria"].isin(_EMERGENCY_CATEGORIAS)
    )
    return ipress_df[mask].copy()


def _print_step(msg: str) -> None:
    print(f"  >> {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Assign health facilities to districts
# ---------------------------------------------------------------------------

def assign_facilities_to_districts(
    ipress_df: pd.DataFrame,
    distritos_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Spatial join between IPRESS health facilities and district boundaries.

    Each facility point is matched to the district polygon it falls inside.
    Facilities that do not intersect any district (e.g. points on boundaries)
    are retained with NaN district attributes (left join).

    Parameters
    ----------
    ipress_df : pd.DataFrame
        Output of clean_ipress(). Must have 'lon' and 'lat' columns.
    distritos_gdf : gpd.GeoDataFrame
        Output of clean_distritos(). CRS must be EPSG:4326.

    Returns
    -------
    gpd.GeoDataFrame
        IPRESS points with district attributes appended.
        Saved to data/processed/ipress_with_districts.gpkg.
    """
    print("\n[assign_facilities_to_districts]")
    _print_step(f"Input: {len(ipress_df):,} facilities, {len(distritos_gdf):,} districts")

    # Convert IPRESS to GeoDataFrame
    ipress_gdf = _ipress_to_geodataframe(ipress_df)

    # Keep only the district columns needed downstream; avoid column collisions
    dist_cols = ["iddist", "distrito", "departamen", "provincia", "geometry"]
    districts  = distritos_gdf[dist_cols].copy()

    # Spatial join — point-in-polygon, left join to retain all facilities
    _print_step("Reprojecting to EPSG:4326 and running point-in-polygon join …")
    joined = gpd.sjoin(
        ipress_gdf,
        districts,
        how="left",
        predicate="within",
    ).drop(columns="index_right", errors="ignore")

    matched   = joined["iddist"].notna().sum()
    unmatched = joined["iddist"].isna().sum()
    _print_step(f"Matched: {matched:,} | Unmatched (no district): {unmatched:,}")

    # Save as GeoPackage (supports long column names, unlike Shapefile)
    out_path = _processed("ipress_with_districts.gpkg")
    joined.to_file(out_path, driver="GPKG")
    _print_step(f"Saved -> {out_path}")

    return joined


# ---------------------------------------------------------------------------
# 2. Assign populated places to districts
# ---------------------------------------------------------------------------

def assign_ccpp_to_districts(
    ccpp_gdf: gpd.GeoDataFrame,
    distritos_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Spatial join between IGN populated places (CCPP) and district boundaries.

    Parameters
    ----------
    ccpp_gdf : gpd.GeoDataFrame
        Output of clean_ccpp(). Point geometry, CRS EPSG:4326.
    distritos_gdf : gpd.GeoDataFrame
        Output of clean_distritos(). CRS EPSG:4326.

    Returns
    -------
    gpd.GeoDataFrame
        CCPP points with district attributes appended.
        Saved to data/processed/ccpp_with_districts.gpkg.
    """
    print("\n[assign_ccpp_to_districts]")
    _print_step(f"Input: {len(ccpp_gdf):,} populated places, {len(distritos_gdf):,} districts")

    dist_cols = ["iddist", "distrito", "departamen", "provincia", "geometry"]
    districts  = distritos_gdf[dist_cols].copy()

    _print_step("Running point-in-polygon join …")
    joined = gpd.sjoin(
        ccpp_gdf,
        districts,
        how="left",
        predicate="within",
    ).drop(columns="index_right", errors="ignore")

    matched   = joined["iddist"].notna().sum()
    unmatched = joined["iddist"].isna().sum()
    _print_step(f"Matched: {matched:,} | Unmatched: {unmatched:,}")

    out_path = _processed("ccpp_with_districts.gpkg")
    joined.to_file(out_path, driver="GPKG")
    _print_step(f"Saved -> {out_path}")

    return joined


# ---------------------------------------------------------------------------
# 3. Nearest emergency facility distance per populated place
# ---------------------------------------------------------------------------

def compute_nearest_facility_distance(
    ccpp_gdf: gpd.GeoDataFrame,
    ipress_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    For each populated place, find the straight-line distance in kilometres
    to the nearest health facility with emergency services.

    Emergency facilities are defined as:
      - tipo == 'ESTABLECIMIENTO DE SALUD CON INTERNAMIENTO', OR
      - categoria in {II-1, II-2, III-1, III-2, II-E, III-E}

    Distance is computed after reprojecting both layers to EPSG:32718
    (WGS 84 / UTM Zone 18S, units: metres). Results are stored in the
    'dist_km' column of the returned GeoDataFrame, which retains the
    original EPSG:4326 CRS.

    Parameters
    ----------
    ccpp_gdf : gpd.GeoDataFrame
        Output of clean_ccpp() or assign_ccpp_to_districts().
    ipress_df : pd.DataFrame
        Output of clean_ipress(). Must have 'lon' and 'lat' columns.

    Returns
    -------
    gpd.GeoDataFrame
        ccpp_gdf with columns added:
          - 'nearest_fac'   : codigo_unico of the nearest emergency facility
          - 'nearest_name'  : name of the nearest emergency facility
          - 'dist_km'       : straight-line distance in kilometres
        Saved to data/processed/ccpp_nearest_facility.gpkg.
    """
    print("\n[compute_nearest_facility_distance]")

    # Build GeoDataFrame of emergency facilities only
    emerg_df  = _filter_emergency_facilities(ipress_df)
    emerg_gdf = _ipress_to_geodataframe(emerg_df)
    _print_step(
        f"Emergency facilities: {len(emerg_gdf):,} "
        f"(filtered from {len(ipress_df):,} total)"
    )
    _print_step(f"Populated places    : {len(ccpp_gdf):,}")

    # Reproject both layers to UTM 18S for metre-accurate distances
    _print_step("Reprojecting to EPSG:32718 (UTM Zone 18S) …")
    ccpp_utm  = ccpp_gdf.to_crs(_CRS_PROJ)
    emerg_utm = emerg_gdf[["codigo_unico", "nombre_del_establecimiento", "geometry"]].to_crs(_CRS_PROJ)

    # sjoin_nearest — O(n log n) via spatial index; returns nearest match per row
    _print_step("Running sjoin_nearest (this may take ~30 s for 136 K points) …")
    result_utm = gpd.sjoin_nearest(
        ccpp_utm,
        emerg_utm,
        how="left",
        distance_col="dist_m",
    ).drop(columns="index_right", errors="ignore")

    # Convert distance to kilometres and rename joined columns
    result_utm["dist_km"] = (result_utm["dist_m"] / 1000).round(3)
    result_utm = result_utm.rename(columns={
        "codigo_unico":                   "nearest_fac",
        "nombre_del_establecimiento":     "nearest_name",
    })

    # Reproject back to geographic CRS for storage
    result = result_utm.to_crs(_CRS_GEO)

    dist_stats = result["dist_km"].describe()
    _print_step(
        f"Distance stats (km) — "
        f"min: {dist_stats['min']:.1f} | "
        f"mean: {dist_stats['mean']:.1f} | "
        f"median: {result['dist_km'].median():.1f} | "
        f"max: {dist_stats['max']:.1f}"
    )

    out_path = _processed("ccpp_nearest_facility.gpkg")
    result.to_file(out_path, driver="GPKG")
    _print_step(f"Saved -> {out_path}")

    return result


# ---------------------------------------------------------------------------
# 4. Build district-level summary GeoDataFrame
# ---------------------------------------------------------------------------

def build_district_geodataframe(
    distritos_gdf: gpd.GeoDataFrame,
    ipress_districts: gpd.GeoDataFrame,
    ccpp_districts: gpd.GeoDataFrame,
    ccpp_distances: gpd.GeoDataFrame,
    emergencias_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Aggregate all spatial layers into a single district-level GeoDataFrame.

    Output columns
    --------------
    iddist          district UBIGEO code (6-digit int)
    distrito        district name
    departamen      department name
    provincia       province name
    fac_count       number of IPRESS facilities in the district
    tot_atend       total emergency attendances (sum of nro_total_atendidos)
    ccpp_count      number of IGN populated places in the district
    avg_dist_km     mean distance (km) from each CCPP to its nearest
                    emergency facility
    geometry        district polygon

    Parameters
    ----------
    distritos_gdf    : clean district boundaries (EPSG:4326)
    ipress_districts : output of assign_facilities_to_districts()
    ccpp_districts   : output of assign_ccpp_to_districts()
    ccpp_distances   : output of compute_nearest_facility_distance()
    emergencias_df   : cleaned emergencias DataFrame (has 'ubigeo' and
                       'nro_total_atendidos' columns)

    Returns
    -------
    gpd.GeoDataFrame
        District-level summary. Saved to:
          data/processed/districts_summary.gpkg  (full column names)
          data/processed/districts_summary.shp   (truncated, for QGIS compat)
    """
    print("\n[build_district_geodataframe]")
    _print_step(f"Base districts: {len(distritos_gdf):,}")

    # Normalise the district key to int64 everywhere for consistent merging
    def _to_int(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64")

    base = distritos_gdf[
        ["iddist", "distrito", "departamen", "provincia", "geometry"]
    ].copy()
    base["iddist"] = _to_int(base["iddist"])

    # ------------------------------------------------------------------
    # A. Facility count per district
    # ------------------------------------------------------------------
    _print_step("Aggregating facility counts …")
    fac = (
        ipress_districts
        .assign(iddist=lambda d: _to_int(d["iddist"]))
        .dropna(subset=["iddist"])
        .groupby("iddist", as_index=False)
        .size()
        .rename(columns={"size": "fac_count"})
    )

    # ------------------------------------------------------------------
    # B. Total emergency attendances per district
    #    emergencias.ubigeo == distritos.iddist (both are 6-digit UBIGEO)
    # ------------------------------------------------------------------
    _print_step("Aggregating emergency attendances …")
    em_agg = (
        emergencias_df
        .assign(iddist=lambda d: _to_int(d["ubigeo"]))
        .groupby("iddist", as_index=False)["nro_total_atendidos"]
        .sum()
        .rename(columns={"nro_total_atendidos": "tot_atend"})
    )

    # ------------------------------------------------------------------
    # C. Populated-place count per district
    # ------------------------------------------------------------------
    _print_step("Aggregating CCPP counts …")
    ccpp_cnt = (
        ccpp_districts
        .assign(iddist=lambda d: _to_int(d["iddist"]))
        .dropna(subset=["iddist"])
        .groupby("iddist", as_index=False)
        .size()
        .rename(columns={"size": "ccpp_count"})
    )

    # ------------------------------------------------------------------
    # D. Average distance to nearest emergency facility per district
    #    Merge ccpp_districts (has iddist) with ccpp_distances (has dist_km)
    #    on their shared index (both derived from the same ccpp_clean.shp).
    # ------------------------------------------------------------------
    _print_step("Aggregating average distances …")
    # ccpp_distances may not have iddist yet — bring it in from ccpp_districts
    dist_col  = ccpp_distances[["dist_km"]].copy()
    dist_col  = dist_col.join(
        ccpp_districts[["iddist"]].assign(iddist=lambda d: _to_int(d["iddist"])),
        how="left",
    )
    avg_dist = (
        dist_col
        .dropna(subset=["iddist", "dist_km"])
        .groupby("iddist", as_index=False)["dist_km"]
        .mean()
        .rename(columns={"dist_km": "avg_dist_km"})
        .assign(avg_dist_km=lambda d: d["avg_dist_km"].round(3))
    )

    # ------------------------------------------------------------------
    # E. Merge all aggregations into the district base
    # ------------------------------------------------------------------
    _print_step("Merging aggregations into district base …")
    result = base.copy()
    for agg_df in [fac, em_agg, ccpp_cnt, avg_dist]:
        result = result.merge(agg_df, on="iddist", how="left")

    # Fill count columns with 0 where no data (districts with no matches)
    result["fac_count"]  = result["fac_count"].fillna(0).astype(int)
    result["tot_atend"]  = result["tot_atend"].fillna(0).astype(int)
    result["ccpp_count"] = result["ccpp_count"].fillna(0).astype(int)
    # avg_dist_km stays NaN where no CCPP points were matched

    n_no_fac  = (result["fac_count"]  == 0).sum()
    n_no_ccpp = (result["ccpp_count"] == 0).sum()
    n_no_dist = result["avg_dist_km"].isna().sum()
    _print_step(f"Districts with 0 facilities : {n_no_fac:,}")
    _print_step(f"Districts with 0 CCPP       : {n_no_ccpp:,}")
    _print_step(f"Districts with no avg dist  : {n_no_dist:,}")

    _print_step(
        f"avg_dist_km summary — "
        f"min: {result['avg_dist_km'].min():.1f} km | "
        f"mean: {result['avg_dist_km'].mean():.1f} km | "
        f"max: {result['avg_dist_km'].max():.1f} km"
    )

    result = gpd.GeoDataFrame(result, geometry="geometry", crs=_CRS_GEO)

    # Save as GeoPackage (preserves all column names)
    gpkg_path = _processed("districts_summary.gpkg")
    result.to_file(gpkg_path, driver="GPKG")
    _print_step(f"Saved -> {gpkg_path}")

    # Also save as Shapefile for QGIS compatibility
    # (column names truncated to 10 chars automatically by geopandas)
    shp_path = _processed("districts_summary.shp")
    result.to_file(shp_path, encoding="utf-8")
    _print_step(f"Saved -> {shp_path}  (shapefile, col names truncated to 10 chars)")

    return result


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))

    from data_loader import load_ipress, load_ccpp, load_distritos
    from cleaning    import clean_ipress, clean_ccpp, clean_distritos

    print("=== Loading and cleaning inputs ===\n")
    ipress_raw    = load_ipress()
    distritos_raw = load_distritos()
    ccpp_raw      = load_ccpp()

    ipress    = clean_ipress(ipress_raw)
    distritos = clean_distritos(distritos_raw)
    ccpp      = clean_ccpp(ccpp_raw)

    emergencias = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                     "emergencias_clean.csv"),
        low_memory=False,
    )

    print("\n=== Running geospatial pipeline ===")

    ipress_dist  = assign_facilities_to_districts(ipress, distritos)
    ccpp_dist    = assign_ccpp_to_districts(ccpp, distritos)
    ccpp_near    = compute_nearest_facility_distance(ccpp, ipress)
    summary      = build_district_geodataframe(
                       distritos, ipress_dist, ccpp_dist, ccpp_near, emergencias
                   )

    print("\n=== District summary sample ===")
    print(summary[["iddist", "distrito", "fac_count", "tot_atend",
                    "ccpp_count", "avg_dist_km"]].head(10).to_string(index=False))
    print(f"\nFinal GeoDataFrame: {len(summary):,} districts × {len(summary.columns)} columns")
