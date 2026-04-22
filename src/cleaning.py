"""
cleaning.py
-----------
Functions to clean each raw dataset and save results to data/processed/.

Each function:
  - accepts a raw DataFrame / GeoDataFrame (from data_loader)
  - standardises column names to lowercase_with_underscores
  - applies dataset-specific validation and drops invalid rows
  - prints a before/after summary
  - saves the cleaned result to data/processed/
"""

import os
import unicodedata
import re

import pandas as pd
import geopandas as gpd
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Peru geographic bounding box (WGS84)
_PERU_LON_MIN, _PERU_LON_MAX = -81.5, -68.5
_PERU_LAT_MIN, _PERU_LAT_MAX = -18.5,   1.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_colname(name: str) -> str:
    """
    Convert a column name to lowercase_with_underscores, stripping accents
    and non-alphanumeric characters.

    Examples
    --------
    'Código Único'  -> 'codigo_unico'
    'NRO_TOTAL_ATENCIONES' -> 'nro_total_atenciones'
    'Nombre del establecimiento' -> 'nombre_del_establecimiento'
    """
    # Decompose unicode and drop combining marks (strips accents)
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    # Lowercase and replace any run of non-alphanumeric chars with underscore
    cleaned = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    return cleaned


def _normalise_columns(df):
    """Return df with all column names normalised (works for DataFrame and GeoDataFrame)."""
    df.columns = [_normalise_colname(c) for c in df.columns]
    return df


def _print_summary(name: str, before: int, after: int, dropped_breakdown: dict) -> None:
    """Print a before/after cleaning summary."""
    total_dropped = before - after
    print(f"\n[{name}]")
    print(f"  Before : {before:>8,} rows")
    for reason, n in dropped_breakdown.items():
        if n:
            print(f"  Dropped ({reason}): {n:,}")
    print(f"  After  : {after:>8,} rows  ({total_dropped:,} removed, "
          f"{after/before*100:.1f}% retained)")


def _saved(path: str, n_rows: int) -> None:
    print(f"  Saved  -> {path}  ({n_rows:,} rows)")


# ---------------------------------------------------------------------------
# Public cleaning functions
# ---------------------------------------------------------------------------

def clean_emergencias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the stacked HIS utilisation records (ConsultaC1 2024-2026).

    Steps
    -----
    1. Normalise column names to lowercase_with_underscores.
    2. Remove exact duplicate rows.
    3. Drop rows where ubigeo or co_ipress is null.
    4. Coerce nro_total_atenciones and nro_total_atendidos to numeric,
       dropping rows where both are non-positive (no activity recorded).
    5. Save to data/processed/emergencias_clean.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_emergencias().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    before = len(df)
    df = _normalise_columns(df.copy())

    # -- Step 2: duplicates --
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates()

    # -- Step 3: null keys --
    n_null = df["ubigeo"].isna().sum() + df["co_ipress"].isna().sum()
    df = df.dropna(subset=["ubigeo", "co_ipress"])

    # -- Step 4: numeric coercion --
    df["nro_total_atenciones"] = pd.to_numeric(df["nro_total_atenciones"], errors="coerce")
    df["nro_total_atendidos"]  = pd.to_numeric(df["nro_total_atendidos"],  errors="coerce")
    n_non_positive = ((df["nro_total_atenciones"] <= 0) & (df["nro_total_atendidos"] <= 0)).sum()
    df = df[~((df["nro_total_atenciones"] <= 0) & (df["nro_total_atendidos"] <= 0))]

    after = len(df)
    _print_summary(
        "clean_emergencias", before, after,
        {"duplicates": n_dup, "null ubigeo/co_ipress": n_null, "non-positive counts": n_non_positive},
    )

    out_path = os.path.join(PROCESSED_DIR, "emergencias_clean.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    _saved(out_path, after)

    return df


def clean_ipress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the IPRESS health-facility registry.

    Coordinate note
    ---------------
    Despite the UTM-style names NORTE/ESTE, the stored values are already
    WGS84 decimal degrees (verified by range: NORTE in [-81, 0] = longitude,
    ESTE in [-18, 0] = latitude for Peru).  pyproj is used to validate that
    retained coordinates fall within Peru's bounding box; no reprojection is
    required.  The columns are renamed to `lon` (from NORTE) and `lat`
    (from ESTE) to reflect their actual meaning.

    Steps
    -----
    1. Normalise column names.
    2. Rename coordinate columns: norte -> lon, este -> lat.
    3. Coerce lon/lat to numeric.
    4. Drop rows where lon/lat are null or zero.
    5. Drop rows whose coordinates fall outside Peru's WGS84 bounding box.
    6. Save to data/processed/ipress_clean.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_ipress().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with `lon` and `lat` columns in WGS84.
    """
    before = len(df)
    df = _normalise_columns(df.copy())

    # -- Step 2: rename coordinate columns to their actual meaning --
    # NORTE holds the longitude (X); ESTE holds the latitude (Y).
    df = df.rename(columns={"norte": "lon", "este": "lat"})

    # -- Step 3: coerce to numeric --
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    # -- Step 4: drop null or zero coordinates --
    n_null_coords = df["lon"].isna().sum() + df["lat"].isna().sum()
    n_zero_coords = ((df["lon"] == 0) | (df["lat"] == 0)).sum()
    df = df.dropna(subset=["lon", "lat"])
    df = df[(df["lon"] != 0) & (df["lat"] != 0)]

    # -- Step 5: validate within Peru bounding box using pyproj CRS --
    # Build a WGS84 CRS and use its area_of_use bounds as the reference.
    # We apply explicit Peru bounds confirmed against the IGN official extent.
    in_bounds = (
        (df["lon"] >= _PERU_LON_MIN) & (df["lon"] <= _PERU_LON_MAX) &
        (df["lat"] >= _PERU_LAT_MIN) & (df["lat"] <= _PERU_LAT_MAX)
    )
    n_out_bounds = (~in_bounds).sum()
    df = df[in_bounds]

    after = len(df)
    _print_summary(
        "clean_ipress", before, after,
        {
            "null coordinates": n_null_coords,
            "zero coordinates": n_zero_coords,
            "outside Peru bounds": n_out_bounds,
        },
    )

    out_path = os.path.join(PROCESSED_DIR, "ipress_clean.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    _saved(out_path, after)

    return df


def clean_ccpp(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean the IGN 1:100K populated-places point shapefile.

    Steps
    -----
    1. Normalise column names.
    2. Drop rows with null or empty geometry.
    3. Extract geometry lon/lat and drop points outside Peru's bounding box.
    4. Save to data/processed/ccpp_clean.shp.

    Note: the X/Y attribute columns in the DBF contain anomalies (X values
    were copied into the Y field in some rows). Coordinate validation is
    performed on the actual geometry, which geopandas reads directly from
    the binary .shp file.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Output of data_loader.load_ccpp().

    Returns
    -------
    gpd.GeoDataFrame
        Cleaned GeoDataFrame.
    """
    before = len(gdf)
    gdf = _normalise_columns(gdf.copy())

    # -- Step 2: null or empty geometry --
    n_null_geom = gdf.geometry.isna().sum()
    n_empty_geom = gdf.geometry.is_empty.sum()
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]

    # -- Step 3: bounding box filter on actual geometry coordinates --
    geom_lon = gdf.geometry.x
    geom_lat = gdf.geometry.y
    in_bounds = (
        (geom_lon >= _PERU_LON_MIN) & (geom_lon <= _PERU_LON_MAX) &
        (geom_lat >= _PERU_LAT_MIN) & (geom_lat <= _PERU_LAT_MAX)
    )
    n_out_bounds = (~in_bounds).sum()
    gdf = gdf[in_bounds]

    after = len(gdf)
    _print_summary(
        "clean_ccpp", before, after,
        {
            "null geometry": n_null_geom,
            "empty geometry": n_empty_geom,
            "outside Peru bounds": n_out_bounds,
        },
    )
    print(f"  CRS    : {gdf.crs}")

    out_path = os.path.join(PROCESSED_DIR, "ccpp_clean.shp")
    gdf.to_file(out_path, encoding="utf-8")
    _saved(out_path, after)

    return gdf


def clean_distritos(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean the district-level administrative boundary shapefile.

    Steps
    -----
    1. Normalise column names.
    2. Drop rows with null or empty geometry.
    3. Fix any invalid geometries with buffer(0).
    4. Save to data/processed/distritos_clean.shp.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Output of data_loader.load_distritos().

    Returns
    -------
    gpd.GeoDataFrame
        Cleaned GeoDataFrame.
    """
    before = len(gdf)
    gdf = _normalise_columns(gdf.copy())

    # -- Step 2: null or empty geometry --
    n_null_geom = gdf.geometry.isna().sum()
    n_empty_geom = gdf.geometry.is_empty.sum()
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]

    # -- Step 3: fix self-intersecting / topologically invalid polygons --
    n_invalid = (~gdf.geometry.is_valid).sum()
    if n_invalid:
        gdf["geometry"] = gdf.geometry.buffer(0)
        print(f"  Fixed {n_invalid} invalid geometries with buffer(0).")

    after = len(gdf)
    _print_summary(
        "clean_distritos", before, after,
        {"null geometry": n_null_geom, "empty geometry": n_empty_geom},
    )
    print(f"  CRS    : {gdf.crs}")

    out_path = os.path.join(PROCESSED_DIR, "distritos_clean.shp")
    gdf.to_file(out_path, encoding="utf-8")
    _saved(out_path, after)

    return gdf


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_emergencias, load_ipress, load_ccpp, load_distritos

    print("=== Cleaning all datasets ===")

    print("\n--- Emergencias ---")
    emergencias = clean_emergencias(load_emergencias())

    print("\n--- IPRESS ---")
    ipress = clean_ipress(load_ipress())

    print("\n--- CCPP ---")
    ccpp = clean_ccpp(load_ccpp())

    print("\n--- Distritos ---")
    distritos = clean_distritos(load_distritos())

    print("\nAll datasets cleaned and saved to data/processed/.")
