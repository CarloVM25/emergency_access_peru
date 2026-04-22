"""
data_loader.py
--------------
Functions to load each raw dataset into memory with correct encoding,
separator, and type handling. Each function prints a confirmation line
with shape and key columns on success.
"""

import os
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

_CONSULTA_FILES = [
    ("ConsultaC1_2024_v22.csv", "latin-1"),
    ("ConsultaC1_2025_v20.csv", "latin-1"),
    ("ConsultaC1_2026_v5.csv",  "utf-8"),
]

_IPRESS_FILE   = "IPRESS.csv"
_CCPP_FILE     = "CCPP_IGN100K.shp"
_DISTRITOS_FILE = "DISTRITOS.shp"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw(filename: str) -> str:
    """Return the absolute path to a file inside data/raw/."""
    return os.path.join(RAW_DIR, filename)


def _confirm(name: str, df, key_cols: list[str]) -> None:
    """Print a one-line confirmation with shape and selected column names."""
    present = [c for c in key_cols if c in df.columns]
    print(
        f"[OK] {name}: {df.shape[0]:,} rows × {df.shape[1]} cols | "
        f"key cols: {present}"
    )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_emergencias() -> pd.DataFrame:
    """
    Stack the three ConsultaC1 CSVs (2024, 2025, 2026) into one DataFrame.

    Each file uses a semicolon separator. Encoding differs across years:
    2024 and 2025 are latin-1; 2026 is UTF-8.

    Returns
    -------
    pd.DataFrame
        Combined HIS utilisation records with a 'source_file' column
        indicating the origin file of each row.
    """
    KEY_COLS = ["ANHO", "MES", "UBIGEO", "CO_IPRESS", "SEXO", "EDAD",
                "NRO_TOTAL_ATENCIONES", "NRO_TOTAL_ATENDIDOS"]

    chunks = []
    for filename, encoding in _CONSULTA_FILES:
        path = _raw(filename)
        df = pd.read_csv(
            path,
            sep=";",
            encoding=encoding,
            low_memory=False,
            on_bad_lines="skip",
        )
        df["source_file"] = filename
        chunks.append(df)
        print(f"  loaded {filename}: {len(df):,} rows")

    combined = pd.concat(chunks, ignore_index=True)

    _confirm("load_emergencias", combined, KEY_COLS)
    return combined


def load_ipress() -> pd.DataFrame:
    """
    Load the IPRESS master registry of health facilities.

    The file is comma-separated with latin-1 encoding. Column names
    contain accented characters that are preserved on read.

    Returns
    -------
    pd.DataFrame
        One row per health facility with location and classification data.
    """
    KEY_COLS = ["Código Único", "Nombre del establecimiento", "UBIGEO",
                "Categoría", "Estado", "NORTE", "ESTE"]

    df = pd.read_csv(
        _raw(_IPRESS_FILE),
        sep=",",
        encoding="latin-1",
        low_memory=False,
        on_bad_lines="skip",
    )

    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    _confirm("load_ipress", df, KEY_COLS)
    return df


def load_ccpp() -> gpd.GeoDataFrame:
    """
    Load the IGN 1:100K populated-places shapefile (point geometry).

    The DBF attribute table is stored in latin-1. The CRS is read
    directly from the accompanying .prj file.

    Returns
    -------
    gpd.GeoDataFrame
        One point per populated place with name, category, and
        administrative codes.
    """
    KEY_COLS = ["NOM_POBLAD", "CATEGORIA", "CAT_POBLAD", "DEP", "PROV", "DIST", "X", "Y"]

    gdf = gpd.read_file(_raw(_CCPP_FILE), encoding="latin-1")

    # Normalise column names: strip whitespace
    gdf.columns = gdf.columns.str.strip()

    _confirm("load_ccpp", gdf, KEY_COLS)
    print(f"       CRS: {gdf.crs} | geometry: {gdf.geom_type.unique().tolist()}")
    return gdf


def load_distritos() -> gpd.GeoDataFrame:
    """
    Load the district-level administrative boundary shapefile (polygon geometry).

    Returns
    -------
    gpd.GeoDataFrame
        One polygon per district with department, province, and district codes.
    """
    KEY_COLS = ["IDDPTO", "DEPARTAMEN", "IDPROV", "PROVINCIA", "IDDIST", "DISTRITO", "CODCCPP", "AREA"]

    gdf = gpd.read_file(_raw(_DISTRITOS_FILE), encoding="latin-1")

    gdf.columns = gdf.columns.str.strip()

    _confirm("load_distritos", gdf, KEY_COLS)
    print(f"       CRS: {gdf.crs} | geometry: {gdf.geom_type.unique().tolist()}")
    return gdf


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Loading all raw datasets ===\n")
    emergencias = load_emergencias()
    print()
    ipress      = load_ipress()
    print()
    ccpp        = load_ccpp()
    print()
    distritos   = load_distritos()
    print("\nAll datasets loaded successfully.")
