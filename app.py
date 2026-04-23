"""
app.py — Emergency Access Peru
================================
Streamlit dashboard presenting the district-level emergency access index
analysis for Peru (2024-2026).

Run from the project root:
    streamlit run app.py
"""

import os
import textwrap

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Page config  (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Emergency Access Peru",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(ROOT, "output", "figures")
TABLES_DIR  = os.path.join(ROOT, "output", "tables")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CAT_ORDER = ["underserved", "below_average", "above_average", "well_served"]
CAT_LABELS = {
    "underserved":   "Underserved",
    "below_average": "Below average",
    "above_average": "Above average",
    "well_served":   "Well served",
}
CAT_COLORS = {
    "underserved":   "#d62728",
    "below_average": "#ff7f0e",
    "above_average": "#4e9ac4",
    "well_served":   "#2ca02c",
}

# ---------------------------------------------------------------------------
# Data loading  (cached so it runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_scores() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(TABLES_DIR, "district_scores.csv"))
    df["baseline_category"] = pd.Categorical(
        df["baseline_category"], categories=CAT_ORDER, ordered=True
    )
    df["alt_category"] = pd.Categorical(
        df["alt_category"], categories=CAT_ORDER, ordered=True
    )
    df["category_changed"] = df["baseline_category"] != df["alt_category"]
    return df


@st.cache_data
def load_html(filename: str) -> str:
    path = os.path.join(FIGURES_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def fig_path(filename: str) -> str:
    return os.path.join(FIGURES_DIR, filename)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cat_badge(cat: str) -> str:
    """Return an HTML colour-coded badge for a category string."""
    color = CAT_COLORS.get(cat, "#888")
    label = CAT_LABELS.get(cat, cat)
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.82em;font-weight:600;">{label}</span>'
    )


def metric_row(cols, labels, values, deltas=None):
    """Render a row of st.metric cards."""
    for i, col in enumerate(cols):
        delta = deltas[i] if deltas else None
        col.metric(labels[i], values[i], delta)


def section_header(text: str, level: int = 3):
    st.markdown(f"{'#' * level} {text}")


# ---------------------------------------------------------------------------
# Load data once at startup
# ---------------------------------------------------------------------------

scores = load_scores()

# ---------------------------------------------------------------------------
# Top-level header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="background:linear-gradient(90deg,#1a5276,#2980b9);
                padding:22px 28px;border-radius:8px;margin-bottom:8px;">
      <h1 style="color:white;margin:0;font-size:2rem;">
        🏥 Emergency Healthcare Access in Peru
      </h1>
      <p style="color:#d6eaf8;margin:6px 0 0;font-size:1rem;">
        District-level analysis of 1,873 districts · 2024–2026 · 683K emergency records
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top KPI strip
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Districts analysed",    f"{len(scores):,}")
k2.metric("Underserved districts", f"{(scores['baseline_category']=='underserved').sum():,}",
          f"{(scores['baseline_category']=='underserved').sum()/len(scores)*100:.0f}% of total")
k3.metric("Mean distance to ER",   f"{scores['avg_dist_km'].mean():.1f} km")
k4.metric("Max distance to ER",    f"{scores['avg_dist_km'].max():.0f} km")
k5.metric("Districts w/ 0 facilities", f"{(scores['fac_count']==0).sum():,}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data & Methodology",
    "📊 Static Analysis",
    "🗺️ GeoSpatial Results",
    "🔍 Interactive Exploration",
])

# ===========================================================================
# TAB 1 — Data & Methodology
# ===========================================================================

with tab1:

    st.markdown("## Problem Statement")
    st.markdown(
        """
        Geographic access to emergency healthcare is a critical determinant of health outcomes,
        yet Peru's 1,873 districts show extreme disparities.  While Lima concentrates most
        specialist hospitals, rural districts in Loreto, Ucayali, and the Andean highlands
        can be **over 200 km from the nearest emergency-capable facility** — a journey that,
        without road access, may take days.

        This analysis builds a quantitative **emergency access index** from four administrative
        datasets to answer:

        1. Which districts are most underserved and where are they?
        2. Do districts with more facilities actually serve more patients?
        3. Is geographic remoteness systematically different across access categories?
        4. How sensitive are district rankings to the choice of index weights?
        """
    )

    st.divider()

    # --- Data sources ---
    st.markdown("## Data Sources")

    data_table = pd.DataFrame([
        {
            "Dataset":     "ConsultaC1 HIS 2024–2026",
            "Format":      "CSV (semicolon-delimited)",
            "Records":     "683,023 (raw) → 649,415 (clean)",
            "Description": "Monthly district-level emergency attendance records from "
                           "Peru's Health Information System (HIS-MINSA). "
                           "Provides tot_atend per facility per month.",
            "Source":      "MINSA — HIS / REUNIS open data",
        },
        {
            "Dataset":     "IPRESS Registry",
            "Format":      "CSV",
            "Records":     "20,819 (raw) → 7,953 (clean, with coordinates)",
            "Description": "Official registry of all health establishments "
                           "(Instituciones Prestadoras de Servicios de Salud). "
                           "Includes WGS84 coordinates, category (I-1 → III-E), "
                           "and institutional classification.",
            "Source":      "SUSALUD open data portal",
        },
        {
            "Dataset":     "CCPP IGN 1:100K",
            "Format":      "Shapefile (point)",
            "Records":     "136,587 populated places",
            "Description": "National Institute of Geography (IGN) populated-place "
                           "layer at 1:100,000 scale. Used as the population proxy "
                           "for the facility-per-place ratio and for computing "
                           "nearest-facility distances.",
            "Source":      "IGN Peru — Infraestructura de Datos Espaciales",
        },
        {
            "Dataset":     "DISTRITOS Administrative Boundaries",
            "Format":      "Shapefile (polygon)",
            "Records":     "1,873 districts",
            "Description": "Official district-level administrative boundaries. "
                           "Used as the spatial unit of analysis for all joins "
                           "and the choropleth map.",
            "Source":      "INEI — Shapefiles Peru",
        },
    ])
    st.dataframe(data_table, use_container_width=True, hide_index=True)

    st.divider()

    # --- Cleaning decisions ---
    st.markdown("## Data Cleaning Decisions")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### HIS Emergencias (ConsultaC1)")
        st.markdown(
            """
            - **33,608 duplicate rows removed** (identical across all 14 columns,
              likely caused by repeated monthly uploads in the source system).
            - `nro_total_atenciones` and `nro_total_atendidos` coerced from
              `object` to numeric — the semicolon CSV parser left them as strings.
            - No rows had null `ubigeo` or `co_ipress`; the guard is in place
              for future data refreshes.
            - Three files stacked vertically with a `source_file` column tracking
              the origin year.
            """
        )

        st.markdown("#### IPRESS Facilities")
        st.markdown(
            """
            - **12,866 facilities dropped** (62%) for missing coordinates.
              This is a known data quality issue in the SUSALUD registry —
              the majority are small posts in remote areas.
            - Column `NORTE` holds **longitude** and `ESTE` holds **latitude**
              (WGS84 decimal degrees despite the UTM-style naming). Renamed
              to `lon`/`lat`.
            - 3 facilities with coordinate value = 0 removed.
            - All retained facilities fall within Peru's bounding box
              (lon −81.5 → −68.5, lat −18.5 → 1.0).
            """
        )

    with col_b:
        st.markdown("#### CCPP Populated Places")
        st.markdown(
            """
            - No rows dropped — all 136,587 points have valid geometry and
              fall within Peru's bounding box.
            - The X/Y attribute columns in the shapefile DBF contain anomalies
              (longitude values copied into the Y field for some rows).
              Coordinate validation was performed on the actual geometry read
              from the binary `.shp` file, not the DBF attributes.
            """
        )

        st.markdown("#### DISTRITOS Boundaries")
        st.markdown(
            """
            - No rows dropped — all 1,873 district polygons are valid.
            - A `buffer(0)` pass is applied to fix any topologically invalid
              polygons (none found in this version, but standard practice for
              INEI boundaries).
            - 221 CCPP points (0.16%) fell outside all district polygons
              (borderline points); they were retained in the distance
              calculation but excluded from district-level counts.
            """
        )

    st.divider()

    # --- Methodology ---
    st.markdown("## Index Construction")

    st.markdown(
        """
        The index is a weighted average of three normalised component scores,
        each capturing a different dimension of emergency access:
        """
    )

    mcol1, mcol2, mcol3 = st.columns(3)

    with mcol1:
        st.markdown(
            """
            **Facility score**
            `fac_count ÷ ccpp_count`

            Facilities per populated place — a supply-to-demand proxy.
            Higher = more facilities relative to the number of settlements.
            Addresses the *availability* dimension.
            """
        )

    with mcol2:
        st.markdown(
            """
            **Activity score**
            `tot_atend ÷ fac_count`

            Emergency attendances per facility — a utilisation proxy.
            Addresses the *utilisation* dimension.  In the **alternative spec**,
            `log1p(tot_atend)` is used to reduce the influence of Lima's
            1.27M-attendance facilities on rural rankings.
            """
        )

    with mcol3:
        st.markdown(
            """
            **Distance score**
            `1 ÷ avg_dist_km`

            Inverse of the mean distance (km) from each populated place to
            the nearest **emergency-capable facility** (category II or III,
            or with inpatient beds).  Lower distance = higher score.
            Addresses the *geographic access* dimension.
            """
        )

    st.markdown("#### Weighting Rationale")

    wcol1, wcol2 = st.columns(2)
    with wcol1:
        st.markdown(
            """
            **Baseline (40 / 30 / 30)**

            Equal emphasis on availability and access, with activity as a
            secondary signal.  A 40% weight on facility score reflects the
            primacy of *having* facilities before utilisation is possible.
            """
        )
        bdata = pd.DataFrame({
            "Component": ["Facility score", "Activity score", "Distance score"],
            "Weight":    ["40%", "30%", "30%"],
            "Activity transform": ["Raw ratio", "—", "—"],
        })
        st.dataframe(bdata, hide_index=True, use_container_width=True)

    with wcol2:
        st.markdown(
            """
            **Alternative (25 / 25 / 50)**

            Doubles the distance weight to emphasise *geographic access* — the
            view that in a country with Peru's terrain and road infrastructure,
            where you are matters as much as what exists.  Log-transforms
            activity to prevent Lima from dominating rural rankings.
            """
        )
        adata = pd.DataFrame({
            "Component": ["Facility score", "Activity score", "Distance score"],
            "Weight":    ["25%", "25%", "50%"],
            "Activity transform": ["Raw ratio", "log1p(tot_atend)", "—"],
        })
        st.dataframe(adata, hide_index=True, use_container_width=True)

    st.markdown(
        """
        **Classification:** Districts are assigned to four categories using
        **quartile-based thresholds** (Q1 / Q2 / Q3) computed from each index.
        Fixed 0–25–50–75 thresholds were rejected because the extreme skew
        (median ≈ 0.37, max ≈ 49.8) would place over 80% of districts in a
        single bin.
        """
    )

    st.divider()

    # --- Limitations ---
    st.markdown("## Limitations")

    lim_col1, lim_col2 = st.columns(2)

    with lim_col1:
        st.markdown(
            """
            **Coverage gaps in IPRESS coordinates**
            62% of registered facilities lack geographic coordinates in the
            SUSALUD registry.  These are disproportionately rural primary-care
            posts — meaning the distance score likely *underestimates* the
            true scarcity of nearby emergency services in remote areas.

            **Straight-line distances**
            The nearest-facility calculation uses Euclidean distance in
            EPSG:32718 (UTM Zone 18S).  Actual travel times along Peru's
            road and river network would be substantially longer, especially
            in Loreto (river transport) and the Andes (mountain roads).
            """
        )

    with lim_col2:
        st.markdown(
            """
            **CCPP as population proxy**
            Populated places are used as the denominator for the facility
            score, not actual population counts.  A small hamlet and a
            town of 20,000 both count as one CCPP point.  Districts with
            few but large settlements may appear under-served.

            **Temporal aggregation**
            Emergency attendance records span 2024–2026 (partial year for
            2026).  Seasonal variation (dengue outbreaks, flooding) is
            averaged out but could produce misleading district rankings if
            one period was unusually high.
            """
        )


# ===========================================================================
# TAB 2 — Static Analysis
# ===========================================================================

with tab2:

    CHARTS = [
        {
            "file":    "01_index_distribution.png",
            "title":   "Distribution of Baseline Emergency Access Index",
            "q":       "Q1 — How are scores distributed?",
            "interp":  (
                "The baseline index is **extremely right-skewed**: the median district "
                "scores only 0.37 out of 100, while Lima (49.8) and Callao (48.5) are "
                "clear outliers.  The right panel shows that 97% of Peru's districts "
                "score below 10 — the national emergency access crisis is not the absence "
                "of a few bad districts but a systemic condition affecting the vast majority."
            ),
            "why":     (
                "Two panels were necessary because a single histogram on the 0–50 scale "
                "compresses the entire interquartile range into a single bin. "
                "A box plot was rejected as it would not show the within-quartile density structure."
            ),
        },
        {
            "file":    "02_top20_underserved.png",
            "title":   "Top 20 Most Underserved Districts",
            "q":       "Q2 — Which districts are most at risk?",
            "interp":  (
                "The 20 worst-scoring districts span Arequipa, Lima (rural provinces), "
                "and Puno.  Three districts have a score of 0.00 — they have no "
                "emergency-capable facilities, no recorded attendances, and no CCPP-based "
                "distance estimate (island or border districts). "
                "Notably, **Arequipa and Lima departments** appear repeatedly, showing that "
                "underservice is not exclusively a jungle or Andean problem."
            ),
            "why":     (
                "Horizontal bars accommodate long district names that would overlap on a "
                "vertical axis. Department colouring adds a geographic dimension without "
                "needing a map panel. A treemap would lose the rank ordering."
            ),
        },
        {
            "file":    "03_scatter_facility_vs_attendance.png",
            "title":   "Facility Count vs Emergency Attendances",
            "q":       "Q3 — Do more facilities mean more patients served?",
            "interp":  (
                "The scatter reveals a **weak positive relationship** between facility count "
                "and attendance — but it is driven almost entirely by the Lima / Callao "
                "cluster (top right).  The majority of districts cluster near the origin: "
                "3 facilities, 0 attendances.  This means facility *presence* alone does "
                "not drive utilisation — either population size, road access, or reporting "
                "completeness also matter.  **Well-served districts (green)** are scattered "
                "across the full x-range, suggesting the index is capturing something "
                "beyond raw facility count."
            ),
            "why":     (
                "Log/symlog axes are essential because both variables span 4+ orders of "
                "magnitude. log1p on the y-axis handles the 1,237 zero-attendance districts. "
                "A density plot was rejected because it would obscure the category structure."
            ),
        },
        {
            "file":    "04_boxplot_distance_by_category.png",
            "title":   "Distance to Nearest Emergency Facility by Category",
            "q":       "Q4 — Is remoteness systematically linked to category?",
            "interp":  (
                "Yes — the distance component does differentiate categories, but imperfectly. "
                "**Underserved districts** have a median distance of ~22 km to the nearest "
                "emergency facility, compared to ~14 km for well-served districts. "
                "However, the distributions overlap substantially and both groups contain "
                "extreme outliers.  This confirms that distance is *one* factor in the index "
                "but facility availability and utilisation also play significant roles."
            ),
            "why":     (
                "A box plot with strip overlay shows both the summary statistics (IQR, median) "
                "and the individual district density.  A bar chart of means would hide the "
                "wide spread; a violin plot is harder for non-statistician audiences."
            ),
        },
        {
            "file":    "05_category_comparison.png",
            "title":   "Baseline vs Alternative Category Reassignment",
            "q":       "Q5 — How sensitive are rankings to the weighting choice?",
            "interp":  (
                "**36% of districts (674) change category** when the distance weight is "
                "doubled.  The cross-tabulation reveals the most important shift: "
                "184 districts that were **well-served under baseline** drop to "
                "above-average or lower under the alternative — these are districts with "
                "high facility counts and attendance but long distances (urban districts "
                "surrounded by large, remote hinterlands).  Conversely, 26 underserved "
                "districts improve — likely small districts very close to a hospital "
                "in an adjacent district."
            ),
            "why":     (
                "A 100% stacked bar makes within-category proportions directly comparable "
                "across groups of different sizes. A Sankey diagram would be more expressive "
                "but requires dependencies outside requirements.txt."
            ),
        },
        {
            "file":    "06_choropleth_baseline_index.png",
            "title":   "Choropleth: District-Level Baseline Index",
            "q":       "Q6 — What is the spatial pattern of emergency access?",
            "interp":  (
                "The choropleth confirms the expected coastal / highland / jungle gradient: "
                "**Lima, Callao, Arequipa, and Ica** (Pacific coast) concentrate the highest "
                "scores.  The Amazon basin (Loreto, Ucayali, Madre de Dios) and the southern "
                "Andes (Apurímac, Huancavelica, Puno) show systematic underservice.  "
                "The categorical panel (right) makes cluster boundaries clearer — a contiguous "
                "band of underserved districts stretches from southern Ayacucho to Puno."
            ),
            "why":     (
                "A choropleth is the only chart type that simultaneously shows spatial "
                "clustering and district identity. Two panels (continuous + categorical) "
                "serve different reading purposes: the gradient reveals magnitude; "
                "the categories reveal policy boundaries."
            ),
        },
    ]

    for chart in CHARTS:
        st.markdown(f"### {chart['title']}")
        st.caption(f"**Analytical question:** {chart['q']}")

        img_col, text_col = st.columns([2, 1])
        with img_col:
            st.image(fig_path(chart["file"]), use_container_width=True)
        with text_col:
            st.markdown("**Interpretation**")
            st.markdown(chart["interp"])
            with st.expander("Why this chart type?"):
                st.markdown(chart["why"])

        st.divider()


# ===========================================================================
# TAB 3 — GeoSpatial Results
# ===========================================================================

with tab3:

    st.markdown("## District Map — Baseline Emergency Access Index")
    st.image(fig_path("06_choropleth_baseline_index.png"), use_container_width=True)
    st.caption(
        "Static choropleth (300 DPI). For an interactive version with tooltips, "
        "see the **Interactive Exploration** tab."
    )

    st.divider()

    # --- Top / Bottom 20 table ---
    st.markdown("## Best and Worst Districts")

    view_mode = st.radio(
        "Show:",
        ["Top 20 — best access", "Bottom 20 — worst access", "All districts"],
        horizontal=True,
    )

    display_cols = {
        "distrito":          "District",
        "departamen":        "Department",
        "provincia":         "Province",
        "fac_count":         "Facilities",
        "tot_atend":         "Total attendances",
        "ccpp_count":        "CCPP places",
        "avg_dist_km":       "Avg dist (km)",
        "baseline_index":    "Baseline index",
        "baseline_category": "Baseline category",
        "alt_index":         "Alt index",
        "alt_category":      "Alt category",
    }

    df_display = scores[list(display_cols.keys())].rename(columns=display_cols).copy()

    if view_mode == "Top 20 — best access":
        df_show = df_display.sort_values("Baseline index", ascending=False).head(20)
    elif view_mode == "Bottom 20 — worst access":
        df_show = df_display.sort_values("Baseline index", ascending=True).head(20)
    else:
        df_show = df_display.sort_values("Baseline index", ascending=False)

    st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Baseline index": st.column_config.ProgressColumn(
                "Baseline index", min_value=0, max_value=scores["baseline_index"].max(),
                format="%.2f",
            ),
            "Alt index": st.column_config.ProgressColumn(
                "Alt index", min_value=0, max_value=scores["alt_index"].max(),
                format="%.2f",
            ),
            "Total attendances": st.column_config.NumberColumn(
                "Total attendances", format="%d",
            ),
            "Avg dist (km)": st.column_config.NumberColumn(
                "Avg dist (km)", format="%.1f",
            ),
        },
    )

    st.divider()

    # --- Department summary ---
    st.markdown("## Summary Statistics by Department")

    dept_agg = (
        scores.groupby("departamen", as_index=False)
        .agg(
            districts        = ("iddist",          "count"),
            underserved      = ("baseline_category", lambda x: (x == "underserved").sum()),
            well_served      = ("baseline_category", lambda x: (x == "well_served").sum()),
            mean_index       = ("baseline_index",    "mean"),
            median_index     = ("baseline_index",    "median"),
            mean_dist_km     = ("avg_dist_km",       "mean"),
            total_atend      = ("tot_atend",         "sum"),
            zero_atend_pct   = ("tot_atend",         lambda x: (x == 0).mean() * 100),
        )
        .sort_values("mean_index", ascending=False)
    )
    dept_agg["pct_underserved"] = (dept_agg["underserved"] / dept_agg["districts"] * 100).round(1)
    dept_agg["mean_index"]      = dept_agg["mean_index"].round(2)
    dept_agg["median_index"]    = dept_agg["median_index"].round(2)
    dept_agg["mean_dist_km"]    = dept_agg["mean_dist_km"].round(1)
    dept_agg["zero_atend_pct"]  = dept_agg["zero_atend_pct"].round(1)

    dept_display = dept_agg.rename(columns={
        "departamen":      "Department",
        "districts":       "Districts",
        "underserved":     "Underserved",
        "well_served":     "Well served",
        "pct_underserved": "% underserved",
        "mean_index":      "Mean index",
        "median_index":    "Median index",
        "mean_dist_km":    "Mean dist (km)",
        "total_atend":     "Total attendances",
        "zero_atend_pct":  "% districts w/ 0 atend",
    })

    st.dataframe(
        dept_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mean index": st.column_config.ProgressColumn(
                "Mean index", min_value=0, max_value=dept_agg["mean_index"].max(),
                format="%.2f",
            ),
            "% underserved": st.column_config.NumberColumn(
                "% underserved", format="%.1f%%"
            ),
            "Total attendances": st.column_config.NumberColumn(
                "Total attendances", format="%d"
            ),
        },
    )

    # Mini scorecard for the worst and best departments
    worst_dept = dept_agg.iloc[-1]
    best_dept  = dept_agg.iloc[0]

    sc1, sc2 = st.columns(2)
    with sc1:
        st.error(
            f"**Worst department: {worst_dept['departamen'].title()}**  \n"
            f"Mean index: {worst_dept['mean_index']:.2f} | "
            f"{worst_dept['pct_underserved']:.0f}% underserved districts | "
            f"Mean dist: {worst_dept['mean_dist_km']:.0f} km"
        )
    with sc2:
        st.success(
            f"**Best department: {best_dept['departamen'].title()}**  \n"
            f"Mean index: {best_dept['mean_index']:.2f} | "
            f"{best_dept['pct_underserved']:.0f}% underserved districts | "
            f"Mean dist: {best_dept['mean_dist_km']:.0f} km"
        )


# ===========================================================================
# TAB 4 — Interactive Exploration
# ===========================================================================

with tab4:

    # --- Folium maps ---
    st.markdown("## Interactive Maps")
    st.caption(
        "Maps are rendered from pre-generated HTML files. "
        "Hover over districts or facilities for detail tooltips."
    )

    map_choice = st.selectbox(
        "Select map",
        [
            "Choropleth — baseline index by district",
            "Health facilities — emergency vs primary care",
            "Category change — baseline vs alternative specification",
        ],
    )
    map_files = {
        "Choropleth — baseline index by district":
            "folium_choropleth.html",
        "Health facilities — emergency vs primary care":
            "folium_facilities.html",
        "Category change — baseline vs alternative specification":
            "folium_comparison.html",
    }
    map_descriptions = {
        "Choropleth — baseline index by district": (
            "Districts shaded by baseline_index (YlOrRd gradient). "
            "Hover for district name, both index scores, both categories, "
            "facility count, and average distance."
        ),
        "Health facilities — emergency vs primary care": (
            "Red markers = 633 emergency-capable facilities (category II/III or with inpatient beds). "
            "Blue clusters = 7,320 primary-care facilities — click to zoom and dissolve clusters. "
            "Click any marker for the full facility card."
        ),
        "Category change — baseline vs alternative specification": (
            "Green = 251 districts that improve category under the distance-heavy spec. "
            "Red = 423 districts that worsen. Grey = 1,199 unchanged. "
            "Toggle layers via the control panel."
        ),
    }

    st.info(map_descriptions[map_choice])

    html_content = load_html(map_files[map_choice])
    components.html(html_content, height=580, scrolling=False)

    st.divider()

    # --- District comparator ---
    st.markdown("## District Comparator")
    st.markdown(
        "Select a department and district to compare baseline vs alternative scores "
        "and see all component breakdowns."
    )

    comp_col1, comp_col2 = st.columns([1, 2])

    with comp_col1:
        dept_list = ["(All departments)"] + sorted(scores["departamen"].unique().tolist())
        selected_dept = st.selectbox("Department", dept_list)

        if selected_dept == "(All departments)":
            dist_options = sorted(scores["distrito"].unique().tolist())
        else:
            dist_options = sorted(
                scores.loc[scores["departamen"] == selected_dept, "distrito"].unique().tolist()
            )

        selected_dist = st.selectbox("District", dist_options)

    row = scores[scores["distrito"] == selected_dist].iloc[0]

    with comp_col2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline index",    f"{row['baseline_index']:.2f}",
                  f"Category: {CAT_LABELS.get(row['baseline_category'], row['baseline_category'])}")
        m2.metric("Alternative index", f"{row['alt_index']:.2f}",
                  f"Category: {CAT_LABELS.get(row['alt_category'], row['alt_category'])}")
        m3.metric("Facilities",        f"{int(row['fac_count'])}")
        m4.metric("Avg dist (km)",
                  f"{row['avg_dist_km']:.1f}" if pd.notna(row["avg_dist_km"]) else "N/A")

    # Component score radar / bar breakdown
    comp_data = pd.DataFrame({
        "Component": [
            "Facility score\n(normalised)",
            "Activity score\n(normalised)",
            "Distance score\n(normalised)",
        ],
        "Baseline": [
            row["baseline_facility_score"],
            row["baseline_activity_score"],
            row["baseline_distance_score"],
        ],
        "Alternative": [
            row["alt_facility_score"],
            row["alt_activity_score"],
            row["alt_distance_score"],
        ],
    }).set_index("Component")

    st.markdown("#### Component Scores (0–1 normalised)")
    st.bar_chart(comp_data, use_container_width=True, height=220)

    # Category badge row
    base_cat = row["baseline_category"]
    alt_cat  = row["alt_category"]
    changed  = base_cat != alt_cat
    badge_md = (
        f"**Baseline:** {CAT_LABELS.get(base_cat, base_cat)} &nbsp;|&nbsp; "
        f"**Alternative:** {CAT_LABELS.get(alt_cat, alt_cat)}"
    )
    if changed:
        direction = "improved" if list(CAT_ORDER).index(str(alt_cat)) > list(CAT_ORDER).index(str(base_cat)) else "worsened"
        badge_md += f"&nbsp;&nbsp; ⚠️ **Category {direction} under alternative spec**"
        st.warning(badge_md, icon="🔁")
    else:
        badge_md += "&nbsp;&nbsp; ✅ Same category in both specifications"
        st.success(badge_md)

    # Context: percentile within department
    dept_scores = scores[scores["departamen"] == row["departamen"]]["baseline_index"]
    pct_rank = (dept_scores < row["baseline_index"]).mean() * 100
    n_dept   = len(dept_scores)
    st.caption(
        f"{selected_dist.title()} ranks in the **{pct_rank:.0f}th percentile** "
        f"within {row['departamen'].title()} department ({n_dept} districts)."
    )

    st.divider()

    # --- Department filter table ---
    st.markdown("## Filter Districts by Department")

    filter_dept = st.selectbox(
        "Filter by department",
        ["(All)"] + sorted(scores["departamen"].unique().tolist()),
        key="filter_dept_select",
    )
    filter_cat = st.multiselect(
        "Filter by baseline category",
        options=CAT_ORDER,
        default=CAT_ORDER,
        format_func=lambda x: CAT_LABELS[x],
    )

    mask = scores["baseline_category"].isin(filter_cat)
    if filter_dept != "(All)":
        mask &= scores["departamen"] == filter_dept

    filtered = (
        scores[mask][[
            "distrito", "departamen", "provincia",
            "fac_count", "tot_atend", "avg_dist_km",
            "baseline_index", "baseline_category",
            "alt_index", "alt_category", "category_changed",
        ]]
        .sort_values("baseline_index", ascending=False)
        .rename(columns={
            "distrito":          "District",
            "departamen":        "Department",
            "provincia":         "Province",
            "fac_count":         "Facilities",
            "tot_atend":         "Attendances",
            "avg_dist_km":       "Avg dist (km)",
            "baseline_index":    "Baseline index",
            "baseline_category": "Baseline cat.",
            "alt_index":         "Alt index",
            "alt_category":      "Alt cat.",
            "category_changed":  "Cat. changed?",
        })
    )

    st.caption(f"Showing {len(filtered):,} districts")
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Baseline index": st.column_config.ProgressColumn(
                "Baseline index", min_value=0,
                max_value=scores["baseline_index"].max(), format="%.2f",
            ),
            "Alt index": st.column_config.ProgressColumn(
                "Alt index", min_value=0,
                max_value=scores["alt_index"].max(), format="%.2f",
            ),
            "Attendances": st.column_config.NumberColumn("Attendances", format="%d"),
            "Cat. changed?": st.column_config.CheckboxColumn("Cat. changed?"),
        },
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Data sources: MINSA HIS-REUNIS, SUSALUD IPRESS registry, IGN 1:100K CCPP, INEI DISTRITOS. "
    "Analysis period: 2024–2026. "
    "Built with Streamlit · GeoPandas · Folium · Seaborn."
)
