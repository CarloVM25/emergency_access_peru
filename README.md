# Emergency Healthcare Access in Peru

A district-level analysis of emergency healthcare access across Peru's 1,873 districts, combining administrative health records, facility registries, and geospatial data to build a quantitative access index for 2024–2026.

---

## Table of Contents

1. [What this project does](#what-this-project-does)
2. [Analytical goal](#analytical-goal)
3. [Datasets](#datasets)
4. [Data cleaning](#data-cleaning)
5. [Index construction](#index-construction)
6. [Project structure](#project-structure)
7. [Installation](#installation)
8. [Running the pipeline](#running-the-pipeline)
9. [Running the Streamlit app](#running-the-streamlit-app)
10. [Main findings](#main-findings)
11. [Limitations](#limitations)

---

## What this project does

This project processes four administrative and geospatial datasets from Peruvian public health institutions to produce a district-level **Emergency Access Index** — a composite score that quantifies how well-served each of Peru's 1,873 districts is in terms of emergency healthcare availability, utilisation, and geographic proximity.

The pipeline covers the full data science workflow:

- **Ingestion** of raw CSV and Shapefile data with encoding and projection handling
- **Cleaning** with documented quality decisions and row-level audit logs
- **Geospatial joins** to assign facilities and populated places to administrative districts
- **Nearest-facility distance computation** for 136,587 populated places against 633 emergency-capable facilities, reprojected to UTM Zone 18S (EPSG:32718)
- **Index construction** under two weighting specifications to test sensitivity
- **Visualisation** as static charts (matplotlib / seaborn) and interactive maps (Folium)
- **Dashboard** via a Streamlit app presenting all results with filtering and comparison tools

---

## Analytical goal

Geographic access to emergency healthcare is a critical but underexplored determinant of health outcomes in Peru. While Lima and the Pacific coast concentrate specialist hospitals, rural districts in the Amazon basin and the southern Andes can be more than 200 km from the nearest emergency-capable facility.

This analysis addresses four questions:

1. **Which districts are most underserved**, and are they geographically clustered or dispersed?
2. **Do districts with more facilities actually serve more patients?** Does facility count alone predict utilisation?
3. **Is geographic remoteness systematically linked to access category?** Does the distance component of the index differentiate underserved from well-served districts?
4. **How sensitive are district rankings to the choice of index weights?** What happens when geographic access is given twice the weight?

---

## Datasets

| Dataset | Format | Raw records | Source |
|---|---|---|---|
| **ConsultaC1 HIS 2024–2026** | CSV (semicolon-delimited, 3 files) | 683,023 rows | MINSA — HIS / REUNIS open data |
| **IPRESS facility registry** | CSV | 20,819 rows | SUSALUD open data portal |
| **CCPP IGN 1:100K** | Shapefile (point) | 136,587 populated places | IGN Peru — Infraestructura de Datos Espaciales |
| **DISTRITOS administrative boundaries** | Shapefile (polygon) | 1,873 districts | INEI — Shapefiles Peru |

**ConsultaC1 HIS** contains monthly district-level emergency attendance records from Peru's Health Information System, one file per year (2024, 2025, and partial 2026). Each row represents the total attendances and patients served at one facility in one month, broken down by age and sex.

**IPRESS** is the official registry of all health establishments (*Instituciones Prestadoras de Servicios de Salud*), including geographic coordinates, MINSA category (I-1 through III-E), institutional classification, and operational status.

**CCPP IGN 1:100K** is the National Institute of Geography's populated-place layer. It is used as a population proxy — each point represents a settlement, from a hamlet to a major city — for computing the facility-per-place ratio and for nearest-facility distance calculations.

**DISTRITOS** provides official district-level administrative boundaries and serves as the spatial unit of analysis for all joins and the choropleth map.

---

## Data cleaning

All cleaning logic is in [`src/cleaning.py`](src/cleaning.py). Cleaned files are written to `data/processed/`.

### ConsultaC1 HIS

- **33,608 duplicate rows removed** (identical across all 14 columns). These originate from repeated monthly uploads in the source system and were confirmed exact duplicates.
- `nro_total_atenciones` and `nro_total_atendidos` coerced from `object` to numeric — the semicolon CSV parser preserved them as strings.
- The 2024 and 2025 files are `latin-1` encoded; the 2026 file is UTF-8. Each file is read with its correct encoding before stacking.
- A `source_file` column is added to track the origin year of each row.
- **Final: 649,415 rows (95.1% retained).**

### IPRESS facility registry

- **12,866 facilities dropped** (61.8%) due to missing coordinates. This is a known data quality issue in the SUSALUD registry — facilities without recorded coordinates are disproportionately small rural posts.
- The columns `NORTE` and `ESTE` hold **WGS84 decimal degrees** despite their UTM-style names. After value-range inspection (NORTE ∈ [−81, 0] = Peru's longitude band; ESTE ∈ [−18, 0] = Peru's latitude band), they were renamed to `lon` and `lat`.
- Three facilities with coordinate value = 0 removed.
- Retained facilities validated against Peru's geographic bounding box (lon −81.5 → −68.5, lat −18.5 → 1.0).
- **Final: 7,953 facilities (38.2% retained).**

### CCPP populated places

- No rows dropped — all 136,587 points have valid geometry and fall within Peru's bounding box.
- The X/Y attribute columns in the shapefile DBF contain anomalies (longitude values copied into the Y field for some records). Coordinate validation was performed on the binary `.shp` geometry, not the DBF attributes.

### DISTRITOS administrative boundaries

- No rows dropped — all 1,873 district polygons are valid.
- A `buffer(0)` pass is applied to correct any topologically invalid polygons (none found in this version, but is standard practice for INEI boundaries).

---

## Index construction

All index logic is in [`src/metrics.py`](src/metrics.py). Geospatial joins and distance computations are in [`src/geospatial.py`](src/geospatial.py).

### Component scores

Three normalised scores (0–1, via min-max normalisation) are computed per district:

| Component | Formula | Dimension |
|---|---|---|
| **Facility score** | `fac_count ÷ ccpp_count` | Supply availability |
| **Activity score** | `tot_atend ÷ fac_count` | Utilisation |
| **Distance score** | `1 ÷ avg_dist_km` | Geographic proximity |

`avg_dist_km` is the mean straight-line distance (km) from each IGN populated place in the district to the nearest **emergency-capable facility** — defined as a facility with `tipo == ESTABLECIMIENTO DE SALUD CON INTERNAMIENTO` or `categoria` in {II-1, II-2, III-1, III-2, II-E, III-E}. This computation uses 633 emergency facilities and 136,587 CCPP points, reprojected to **EPSG:32718** (WGS 84 / UTM Zone 18S) for metre-accurate distances via `geopandas.sjoin_nearest`.

Edge cases: districts with `ccpp_count = 0` receive `facility_score = 0`; districts with `fac_count = 0` receive `activity_score = 0`; the 3 districts with no CCPP-matched distances receive `distance_score = 0` (worst case).

### Baseline specification

| Component | Weight | Activity transformation |
|---|---|---|
| Facility score | **40%** | Raw ratio |
| Activity score | **30%** | Raw `tot_atend ÷ fac_count` |
| Distance score | **30%** | `1 ÷ avg_dist_km` |

Facility score receives the highest weight because facility *presence* is a prerequisite for utilisation and access. Activity and distance are given equal secondary weight.

### Alternative specification

| Component | Weight | Activity transformation |
|---|---|---|
| Facility score | **25%** | Raw ratio |
| Activity score | **25%** | `log1p(tot_atend) ÷ fac_count` |
| Distance score | **50%** | `1 ÷ avg_dist_km` |

The distance weight is doubled to emphasise geographic access — the position that in a country with Peru's terrain and road infrastructure, *where you are* matters as much as *what exists*. The log transformation on `tot_atend` prevents Lima's 1.27M-attendance facilities from dominating the activity score and masking rural underservice.

### Classification

Districts are assigned to four categories using **quartile-based thresholds** (Q1 / Q2 / Q3) computed from each specification's index separately. Fixed 0/25/50/75 thresholds were rejected because the index is extremely right-skewed (median ≈ 0.37, max ≈ 49.8 for the baseline) — fixed thresholds would place over 80% of districts in a single bin.

| Category | Threshold |
|---|---|
| **Well served** | Index > Q3 (top 25%) |
| **Above average** | Q2 < index ≤ Q3 |
| **Below average** | Q1 < index ≤ Q2 |
| **Underserved** | Index ≤ Q1 (bottom 25%) |

---

## Project structure

```
emergency_access_peru/
│
├── app.py                        # Streamlit dashboard (4 tabs)
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_loader.py            # Load raw CSVs and shapefiles
│   ├── cleaning.py               # Clean each dataset, save to processed/
│   ├── geospatial.py             # Spatial joins and nearest-facility distances
│   ├── metrics.py                # Baseline and alternative index computation
│   ├── visualization.py          # Static charts (matplotlib) + Folium maps
│   └── utils.py                  # Full pipeline runner
│
├── data/
│   ├── raw/                      # Original source files (git-ignored)
│   └── processed/                # Cleaned outputs (git-ignored)
│
└── output/
    ├── figures/                  # PNG charts and Folium HTML maps
    └── tables/                   # district_scores.csv
```

---

## Installation

Python 3.10+ is required. All dependencies are listed in [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt
```

Key dependencies and their roles:

| Package | Role |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `geopandas`, `shapely`, `pyproj`, `pyogrio` | Geospatial processing |
| `matplotlib`, `seaborn` | Static charts |
| `folium` | Interactive HTML maps |
| `streamlit` | Dashboard |
| `openpyxl` | Excel I/O |

> **Note:** `pyogrio` is the default I/O engine for geopandas ≥ 1.0 and replaces `fiona`. If you are using an older geopandas version, install `fiona` instead.

---

## Running the pipeline

Place the four raw datasets in `data/raw/` following the filenames expected by `src/data_loader.py`:

```
data/raw/
├── ConsultaC1_2024_v22.csv
├── ConsultaC1_2025_v20.csv
├── ConsultaC1_2026_v5.csv
├── IPRESS.csv
├── CCPP_IGN100K.shp  (+ .dbf, .shx, .prj, .cpg)
└── DISTRITOS.shp     (+ .dbf, .shx, .prj, .cpg)
```

Then run the full pipeline from the project root:

```bash
python src/utils.py
```

This executes four steps in order and takes approximately 25–30 seconds on a standard laptop:

| Step | Module | Output |
|---|---|---|
| 1. Load | `data_loader.py` | Raw DataFrames in memory |
| 2. Clean | `cleaning.py` | `data/processed/*.csv`, `data/processed/*.shp` |
| 3. Geospatial | `geospatial.py` | `data/processed/*.gpkg`, `data/processed/districts_summary.gpkg` |
| 4. Metrics | `metrics.py` | `output/tables/district_scores.csv` |

To regenerate the charts and Folium maps after the pipeline has run:

```bash
python src/visualization.py
```

Individual modules can also be run standalone for development:

```bash
python src/cleaning.py       # re-clean only
python src/geospatial.py     # re-run spatial joins only
python src/metrics.py        # re-compute index only
```

---

## Running the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` and presents four tabs:

| Tab | Contents |
|---|---|
| **Data & Methodology** | Problem statement, data sources, cleaning decisions, index construction, limitations |
| **Static Analysis** | All 6 PNG charts with interpretations and chart-type rationale |
| **GeoSpatial Results** | Static choropleth, sortable district table (top/bottom 20), department summary |
| **Interactive Exploration** | 3 embedded Folium maps, district comparator, department/category filter |

The app loads data exclusively from `output/tables/district_scores.csv` and `output/figures/` — the pipeline must have run at least once before launching the app.

---

## Main findings

### Index distribution

The baseline emergency access index is **extremely right-skewed**. The median district scores **0.37 out of a possible 100**, while Lima (49.8) and Bellavista-Callao (48.5) are clear outliers. Ninety-seven percent of Peru's 1,873 districts score below 10. This is not the absence of a few bad districts — it reflects a systemic condition affecting the vast majority of the country.

### Underserved districts

- **435 districts (23.2%)** are classified as underserved under the baseline specification.
- **55 districts have zero IPRESS facilities** with known coordinates.
- The worst-scoring districts include remote Amazonian jurisdictions (Purus, Yaguas, Andoas in Loreto/Ucayali) that combine zero recorded emergency attendances with average distances exceeding **140–228 km** to the nearest emergency-capable facility.
- Underservice is not exclusively a jungle or Andean problem — Arequipa and Lima's rural provinces also appear in the bottom 20.

### Spatial patterns

The choropleth reveals a clear coastal / highland / jungle gradient. Lima, Callao, Arequipa, and Ica (Pacific coast) concentrate the highest scores. A contiguous band of underserved districts stretches from southern Ayacucho through Huancavelica to Puno. The Amazon basin (Loreto, Ucayali, Madre de Dios) shows near-universal underservice driven primarily by extreme distances.

### Facility utilisation

The scatter of facility count versus emergency attendances reveals a **weak positive relationship** driven almost entirely by Lima and Callao. The vast majority of districts have 3 facilities and zero recorded attendances — suggesting that facility presence alone does not drive utilisation, and that reporting completeness, population size, and road access all play a role.

### Geographic distance

Emergency-capable facility distances range from **0.15 km to 228 km**, with a mean of 23.8 km and a median of 18.5 km. Underserved districts have a median distance of approximately 22 km to the nearest emergency facility, compared to ~14 km for well-served districts — confirming that distance is one differentiating factor in the index, though with substantial overlap between categories.

### Specification sensitivity

**674 districts (36.0%) change access category** when the alternative specification doubles the distance weight and log-transforms the activity score. Of these, 251 improve and 423 worsen. The most significant shift: 184 districts classified as *well-served* in the baseline drop to *above-average* or lower in the alternative — these are districts with high facility counts and attendance but long population-weighted distances, reflecting urban districts with large, remote hinterlands. The sensitivity of rankings to weighting choices underscores the importance of transparent methodological documentation when using composite indices for policy decisions.

---

## Limitations

**Coordinate coverage.** 62% of registered IPRESS facilities lack geographic coordinates in the SUSALUD registry. These are disproportionately rural primary-care posts, meaning the distance score likely *underestimates* the true scarcity of nearby emergency services in remote areas. The 633 emergency-capable facilities used for distance calculations represent a lower bound.

**Straight-line distances.** The nearest-facility calculation uses Euclidean distance in EPSG:32718 (UTM Zone 18S). Actual travel times along Peru's road and river network can be substantially longer — especially in Loreto (river transport) and the Andes (mountain roads with switchbacks). A network-distance analysis using Peru's road graph would produce more accurate access estimates.

**CCPP as a population proxy.** Populated places are used as the denominator for the facility score and as the spatial units for distance calculation. A small hamlet and a city of 100,000 both count as one CCPP point. Districts with few but large settlements may appear under-served relative to their actual population burden.

**Temporal aggregation.** Emergency attendance records span 2024–2026 (with 2026 as a partial year). Seasonal variation — dengue outbreaks, flooding, tourist seasons — is averaged out across the period. A district hit by a dengue outbreak in 2024 may have artificially high attendance, inflating its activity score.

**Definition of emergency-capable facilities.** Emergency facilities are defined by MINSA category (II or III) or by the presence of inpatient beds. This excludes some I-4 facilities with 24-hour capacity and includes some III-E specialty hospitals that may not offer general emergency care. A definition based on declared emergency service availability would be more precise but is not available in the public registry.

**Single-year cross-section.** The index is a snapshot of the 2024–2026 period. It does not capture trends in access over time or the impact of infrastructure investments made before or after this window.
