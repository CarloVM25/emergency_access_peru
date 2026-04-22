"""
utils.py
--------
Main pipeline runner for emergency_access_peru.

Executes the full data pipeline in order:
  1. Load all raw datasets   (data_loader.py)
  2. Clean all datasets      (cleaning.py)

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    """Print a clearly visible section header."""
    width = 60
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
    _section("STEP 1 — Load raw datasets")
    t = time.time()

    print("\n[1/4] load_emergencias()")
    emergencias = load_emergencias()

    print("\n[2/4] load_ipress()")
    ipress = load_ipress()

    print("\n[3/4] load_ccpp()")
    ccpp = load_ccpp()

    print("\n[4/4] load_distritos()")
    distritos = load_distritos()

    print(f"\n  All datasets loaded in {_elapsed(t)}.")
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
    _section("STEP 2 — Clean datasets")
    t = time.time()

    print("\n[1/4] clean_emergencias()")
    emergencias = clean_emergencias(raw["emergencias"])

    print("\n[2/4] clean_ipress()")
    ipress = clean_ipress(raw["ipress"])

    print("\n[3/4] clean_ccpp()")
    ccpp = clean_ccpp(raw["ccpp"])

    print("\n[4/4] clean_distritos()")
    distritos = clean_distritos(raw["distritos"])

    print(f"\n  All datasets cleaned in {_elapsed(t)}.")
    return {
        "emergencias": emergencias,
        "ipress":      ipress,
        "ccpp":        ccpp,
        "distritos":   distritos,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> dict:
    """
    Execute the full data pipeline from raw files to cleaned outputs.

    Returns
    -------
    dict
        Cleaned datasets keyed by name.
    """
    pipeline_start = time.time()

    print("\n" + "#" * 60)
    print("#  emergency_access_peru — data pipeline")
    print("#" * 60)

    raw     = run_load()
    cleaned = run_clean(raw)

    _section("PIPELINE COMPLETE")
    print(f"\n  Total time : {_elapsed(pipeline_start)}")
    print("\n  Outputs written to data/processed/:")
    print("    emergencias_clean.csv")
    print("    ipress_clean.csv")
    print("    ccpp_clean.shp")
    print("    distritos_clean.shp")
    print()

    return cleaned


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
