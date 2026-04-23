"""
visualization.py
----------------
Static chart generation for the emergency_access_peru project.

All functions load data from data/processed/ and output/tables/ so they
can run independently of the rest of the pipeline.  Each function answers
one specific analytical question, saves a 300-DPI PNG to output/figures/,
and returns the matplotlib Figure for optional further use.

Analytical questions addressed
-------------------------------
Q1. How are emergency access scores distributed across Peru's 1,873 districts?
Q2. Which specific districts are most at risk and where are they?
Q3. Do districts with more facilities actually serve more patients?
Q4. Is geographic remoteness systematically different across access categories?
Q5. Which districts change category under the alternative (distance-heavy) spec?
Q6. What is the spatial pattern of emergency access across Peru?
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend (no GUI popup)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT         = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(_ROOT, "data",   "processed")
TABLES_DIR    = os.path.join(_ROOT, "output", "tables")
FIGURES_DIR   = os.path.join(_ROOT, "output", "figures")

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

# Category display order (worst to best) used on every chart axis
CAT_ORDER = ["underserved", "below_average", "above_average", "well_served"]
CAT_LABELS = {
    "underserved":   "Underserved",
    "below_average": "Below average",
    "above_average": "Above average",
    "well_served":   "Well served",
}

# Consistent colour per category across all charts
CAT_COLORS = {
    "underserved":   "#d62728",   # red
    "below_average": "#ff7f0e",   # orange
    "above_average": "#4e9ac4",   # steel-blue
    "well_served":   "#2ca02c",   # green
}

DPI       = 300
FONT_BASE = 10
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi":      DPI,
    "savefig.dpi":     DPI,
    "font.size":       FONT_BASE,
    "axes.titlesize":  FONT_BASE + 2,
    "axes.labelsize":  FONT_BASE + 1,
    "legend.fontsize": FONT_BASE - 1,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_path(filename: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, filename)


def _load_scores() -> pd.DataFrame:
    path = os.path.join(TABLES_DIR, "district_scores.csv")
    df = pd.read_csv(path)
    df["baseline_category"] = pd.Categorical(
        df["baseline_category"], categories=CAT_ORDER, ordered=True
    )
    df["alt_category"] = pd.Categorical(
        df["alt_category"], categories=CAT_ORDER, ordered=True
    )
    return df


def _load_geodataframe() -> gpd.GeoDataFrame:
    return gpd.read_file(os.path.join(PROCESSED_DIR, "districts_summary.gpkg"))


def _save(fig: plt.Figure, filename: str) -> str:
    path = _fig_path(filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


def _cat_legend_handles() -> list:
    """Reusable category legend patch list (worst to best)."""
    return [
        mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[c])
        for c in CAT_ORDER
    ]


# ---------------------------------------------------------------------------
# Chart 1 — Distribution of baseline_index  (Q1)
# ---------------------------------------------------------------------------

def plot_index_distribution(scores: pd.DataFrame = None) -> plt.Figure:
    """
    Answers Q1: How are emergency access scores distributed across Peru?

    Chart type — histogram with KDE overlay, two panels:
    The baseline_index is extremely right-skewed (median ~0.37, max ~49.8).
    A single histogram would compress 97% of districts into the first bin.
    Two panels solve this: the left shows the full range (revealing Lima /
    Callao outliers); the right zooms to 0-10 where the bulk of the
    distribution lives.  A KDE line adds distributional shape without
    requiring the reader to mentally smooth the bars.  A box plot was
    considered but would not show within-quartile density structure.
    """
    if scores is None:
        scores = _load_scores()

    idx      = scores["baseline_index"].dropna()
    zoom_max = 10   # covers ~97% of districts

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Distribution of Baseline Emergency Access Index\n"
        "1,873 districts  |  Peru 2024-2026",
        fontsize=FONT_BASE + 3, fontweight="bold", y=1.01,
    )

    # --- Left: full range ---
    ax = axes[0]
    sns.histplot(idx, bins=60, kde=True, color="#4e9ac4",
                 line_kws={"linewidth": 2}, ax=ax)
    ax.axvline(idx.median(), color="crimson",    lw=1.5, ls="--",
               label=f"Median = {idx.median():.2f}")
    ax.axvline(idx.mean(),   color="darkorange", lw=1.5, ls=":",
               label=f"Mean   = {idx.mean():.2f}")
    ax.set_title("Full range (0 - 50)")
    ax.set_xlabel("Baseline index (0-100 scale)")
    ax.set_ylabel("Number of districts")
    ax.legend()

    # --- Right: zoomed ---
    ax = axes[1]
    zoomed    = idx[idx <= zoom_max]
    pct_shown = len(zoomed) / len(idx) * 100
    sns.histplot(zoomed, bins=50, kde=True, color="#4e9ac4",
                 line_kws={"linewidth": 2}, ax=ax)
    ax.axvline(idx.median(), color="crimson", lw=1.5, ls="--",
               label=f"Median = {idx.median():.2f}")

    # Mark quartile thresholds used for classification
    q1, q2, q3 = idx.quantile([0.25, 0.50, 0.75])
    for q, label, color in [
        (q1, f"Q1={q1:.2f}", "#d62728"),
        (q2, f"Q2={q2:.2f}", "#ff7f0e"),
        (q3, f"Q3={q3:.2f}", "#2ca02c"),
    ]:
        if q <= zoom_max:
            ax.axvline(q, color=color, lw=1.2, ls="-.", alpha=0.8, label=label)
    ax.set_title(f"Zoomed: 0 - {zoom_max}  ({pct_shown:.0f}% of districts)")
    ax.set_xlabel("Baseline index (0-100 scale)")
    ax.set_ylabel("Number of districts")
    ax.legend(fontsize=FONT_BASE - 1)

    # Inset: category counts
    cat_counts = scores["baseline_category"].value_counts().reindex(CAT_ORDER)
    lines = [f"{CAT_LABELS[c]}: {cat_counts[c]:,}" for c in CAT_ORDER]
    axes[1].text(
        0.97, 0.97, "\n".join(lines),
        transform=axes[1].transAxes,
        va="top", ha="right", fontsize=FONT_BASE - 1,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    _save(fig, "01_index_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# Chart 2 — Top 20 most underserved districts  (Q2)
# ---------------------------------------------------------------------------

def plot_top20_underserved(scores: pd.DataFrame = None) -> plt.Figure:
    """
    Answers Q2: Which specific districts are most at risk, and in which
    departments are they concentrated?

    Chart type — horizontal bar chart coloured by department:
    A horizontal bar chart is ideal because district names are long strings
    that would overlap on a vertical axis.  Colouring by department adds a
    geographic dimension without requiring a map, making it easy to spot
    whether underservice is concentrated in specific regions.  A treemap
    was considered but loses the rank ordering.  A ranked table was rejected
    because it does not show magnitude differences visually.

    Districts are selected from the 'underserved' category, sorted by
    baseline_index ascending (worst first), with avg_dist_km as the
    secondary sort to break ties among districts with identical index scores.
    """
    if scores is None:
        scores = _load_scores()

    worst20 = (
        scores[scores["baseline_category"] == "underserved"]
        .sort_values(["baseline_index", "avg_dist_km"], ascending=[True, False])
        .head(20)
        .copy()
    )

    # Assign a distinct colour per department (subset may have 5-12 departments)
    depts   = worst20["departamen"].unique()
    palette = sns.color_palette("tab20", len(depts))
    dept_color = dict(zip(depts, palette))
    bar_colors = worst20["departamen"].map(dept_color)

    worst20["label"] = (
        worst20["distrito"].str.title() + "  (" +
        worst20["departamen"].str.title() + ")"
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(
        worst20["label"],
        worst20["baseline_index"],
        color=bar_colors,
        edgecolor="white", linewidth=0.5,
    )

    for bar, val in zip(bars, worst20["baseline_index"]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=FONT_BASE - 1.5,
        )

    ax.set_xlabel("Baseline emergency access index (lower = worse)")
    ax.set_title(
        "Top 20 Most Underserved Districts — Baseline Specification\n"
        "Filtered to 'underserved' category  |  sorted ascending by index",
        fontweight="bold",
    )
    ax.invert_yaxis()   # worst district at the top
    ax.set_xlim(0, worst20["baseline_index"].max() * 1.35)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="x", which="major", alpha=0.4)
    ax.grid(axis="x", which="minor", alpha=0.15)

    legend_handles = [
        mpatches.Patch(color=dept_color[d], label=d.title()) for d in depts
    ]
    ax.legend(
        handles=legend_handles, title="Department",
        bbox_to_anchor=(1.01, 1), loc="upper left",
        borderaxespad=0, frameon=True,
    )

    fig.tight_layout()
    _save(fig, "02_top20_underserved.png")
    return fig


# ---------------------------------------------------------------------------
# Chart 3 — Scatter: facility count vs total attendances  (Q3)
# ---------------------------------------------------------------------------

def plot_scatter_facility_vs_attendance(scores: pd.DataFrame = None) -> plt.Figure:
    """
    Answers Q3: Do districts with more facilities actually serve more patients?
    Does high facility count guarantee high utilisation?

    Chart type — scatter on log-transformed axes, coloured by category:
    Both fac_count and tot_atend are right-skewed (Lima: 44 facilities,
    1.27 M attendances; median: 3 facilities, 0 attendances).  Log
    transformation spreads the dense cluster near the origin.  tot_atend
    uses log1p to handle the 1,237 districts with zero attendances.
    Category colour reveals whether the index classification aligns with
    the underlying facility/attendance relationship.  A 2-D density plot
    was considered but would obscure the category structure.
    """
    if scores is None:
        scores = _load_scores()

    rng = np.random.default_rng(42)
    df  = scores.copy()
    df["log1p_atend"] = np.log1p(df["tot_atend"])
    # Horizontal jitter so fac_count=0 districts don't all stack on x=0
    df["fac_jitter"] = df["fac_count"] + rng.uniform(-0.15, 0.15, len(df))

    fig, ax = plt.subplots(figsize=(11, 7))

    for cat in CAT_ORDER:
        sub = df[df["baseline_category"] == cat]
        ax.scatter(
            sub["fac_jitter"], sub["log1p_atend"],
            c=CAT_COLORS[cat], label=CAT_LABELS[cat],
            alpha=0.5, s=16, linewidths=0,
        )

    # Label the 5 highest-attendance outliers
    for _, row in df.nlargest(5, "tot_atend").iterrows():
        ax.annotate(
            row["distrito"].title(),
            xy=(row["fac_count"], row["log1p_atend"]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=FONT_BASE - 2.5, color="black",
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.7),
        )

    ax.set_xscale("symlog", linthresh=1)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Number of IPRESS facilities  (symlog scale, jittered)")
    ax.set_ylabel("log(1 + total emergency attendances)")
    ax.set_title(
        "Facility Count vs Emergency Attendances by Access Category\n"
        "Each point = 1 district  |  1,237 districts with 0 attendances sit on y = 0",
        fontweight="bold",
    )
    ax.legend(handles=_cat_legend_handles(), title="Baseline category",
              frameon=True, loc="upper left")

    n_zero = (scores["tot_atend"] == 0).sum()
    ax.text(
        0.01, 0.01,
        f"{n_zero:,} districts have tot_atend = 0  (log1p = 0)",
        transform=ax.transAxes, fontsize=FONT_BASE - 2,
        color="grey", va="bottom",
    )

    fig.tight_layout()
    _save(fig, "03_scatter_facility_vs_attendance.png")
    return fig


# ---------------------------------------------------------------------------
# Chart 4 — Box plot: avg_dist_km by category  (Q4)
# ---------------------------------------------------------------------------

def plot_boxplot_distance_by_category(scores: pd.DataFrame = None) -> plt.Figure:
    """
    Answers Q4: Is geographic remoteness systematically different across
    access categories?  Does the distance component actually differentiate
    underserved from well-served districts?

    Chart type — box plot with strip overlay, ordered worst-to-best:
    A box plot concisely shows median, IQR, and outliers per group.  The
    strip (jittered dot) overlay reveals the sample density within each box,
    which matters here because the 'underserved' group contains extreme
    outliers (>200 km).  A violin plot was considered but is harder to read
    for non-statisticians.  A bar chart of means would hide the wide spread
    visible in the IQR boxes.
    """
    if scores is None:
        scores = _load_scores()

    df = scores.dropna(subset=["avg_dist_km"]).copy()
    df["cat_label"] = df["baseline_category"].map(CAT_LABELS)
    label_order     = [CAT_LABELS[c] for c in CAT_ORDER]
    palette         = {CAT_LABELS[c]: CAT_COLORS[c] for c in CAT_ORDER}

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df, x="cat_label", y="avg_dist_km",
        order=label_order, hue="cat_label", hue_order=label_order,
        palette=palette, legend=False,
        width=0.5, fliersize=3, linewidth=1.2, ax=ax,
    )
    sns.stripplot(
        data=df, x="cat_label", y="avg_dist_km",
        order=label_order, hue="cat_label", hue_order=label_order,
        palette=palette, legend=False,
        size=2.5, alpha=0.25, jitter=True, ax=ax,
    )

    # Annotate median per group
    for i, cat in enumerate(CAT_ORDER):
        med = df.loc[df["baseline_category"] == cat, "avg_dist_km"].median()
        ax.text(i, med + 0.8, f"{med:.1f} km",
                ha="center", va="bottom", fontsize=FONT_BASE - 1,
                fontweight="bold")

    # Sample-size labels above each box
    ymax = ax.get_ylim()[1]
    for i, cat in enumerate(CAT_ORDER):
        n = (df["baseline_category"] == cat).sum()
        ax.text(i, ymax * 0.98, f"n={n:,}",
                ha="center", va="top", fontsize=FONT_BASE - 2, color="dimgrey")

    ax.set_xlabel("Baseline access category (worst to best)")
    ax.set_ylabel("Average distance to nearest\nemergency facility (km)")
    ax.set_title(
        "Distance to Nearest Emergency Facility by Access Category\n"
        "Box = IQR  |  whiskers = 1.5x IQR  |  dots = individual districts",
        fontweight="bold",
    )
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, "04_boxplot_distance_by_category.png")
    return fig


# ---------------------------------------------------------------------------
# Chart 5 — Stacked bar: baseline vs alternative category changes  (Q5)
# ---------------------------------------------------------------------------

def plot_category_comparison(scores: pd.DataFrame = None) -> plt.Figure:
    """
    Answers Q5: Which districts change category when the index gives twice
    the weight to geographic distance?  Does emphasising access distance
    surface different at-risk districts than the baseline specification?

    Chart type — 100% stacked bar chart (one bar per baseline category):
    Each bar represents all districts in a given baseline category; its
    coloured segments show where those same districts land under the
    alternative spec.  A 100% stacked bar makes the proportion of movers
    immediately readable and comparable across categories of different sizes.
    A Sankey diagram would be more expressive but requires external
    dependencies not included in requirements.txt.  A grouped bar chart
    shows absolute counts but not the within-category proportion, which is
    what the analytical question requires.
    """
    if scores is None:
        scores = _load_scores()

    # Cross-tabulation: rows = baseline, columns = alternative
    ct = pd.crosstab(scores["baseline_category"], scores["alt_category"])
    ct = ct.reindex(index=CAT_ORDER, columns=CAT_ORDER, fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100   # row-normalised %

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={"width_ratios": [3, 2]})

    # --- Left: 100% stacked bar ---
    ax = axes[0]
    x_labels = [CAT_LABELS[c] for c in CAT_ORDER]
    bottoms  = np.zeros(len(CAT_ORDER))

    for alt_cat in CAT_ORDER:
        heights = ct_pct[alt_cat].values
        bars = ax.bar(
            x_labels, heights, bottom=bottoms,
            color=CAT_COLORS[alt_cat], label=CAT_LABELS[alt_cat],
            edgecolor="white", linewidth=0.6,
        )
        for bar, h, bot in zip(bars, heights, bottoms):
            if h >= 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + h / 2,
                    f"{h:.0f}%",
                    ha="center", va="center",
                    fontsize=FONT_BASE - 1, fontweight="bold", color="white",
                )
        bottoms += heights

    ax.set_ylim(0, 100)
    ax.set_ylabel("Proportion of districts (%)")
    ax.set_xlabel("Baseline category")
    ax.set_title(
        "Category Reassignment Under Alternative Specification\n"
        "Bars = baseline categories  |  segments = alternative assignment",
        fontweight="bold",
    )
    ax.legend(
        handles=_cat_legend_handles(), title="Alternative category",
        bbox_to_anchor=(1.01, 1), loc="upper left",
    )

    # --- Right: summary panel ---
    ax2 = axes[1]
    ax2.axis("off")

    n_total   = len(scores)
    n_changed = (scores["baseline_category"] != scores["alt_category"]).sum()
    n_same    = n_total - n_changed
    changed   = scores[scores["baseline_category"] != scores["alt_category"]]
    n_up      = (changed["alt_category"] > changed["baseline_category"]).sum()
    n_down    = (changed["alt_category"] < changed["baseline_category"]).sum()

    ax2.text(0.05, 0.95, "Specification comparison summary",
             transform=ax2.transAxes, fontsize=FONT_BASE, fontweight="bold", va="top")

    rows = [
        ("Total districts",             f"{n_total:,}"),
        ("Unchanged category",           f"{n_same:,}  ({n_same/n_total*100:.0f}%)"),
        ("Changed category",             f"{n_changed:,}  ({n_changed/n_total*100:.0f}%)"),
        ("  Improved in alt spec",        f"{n_up:,}"),
        ("  Worsened in alt spec",        f"{n_down:,}"),
    ]
    y = 0.82
    for label, value in rows:
        ax2.text(0.05, y, label,  transform=ax2.transAxes,
                 fontsize=FONT_BASE - 0.5, va="top", color="dimgrey")
        ax2.text(0.95, y, value,  transform=ax2.transAxes,
                 fontsize=FONT_BASE - 0.5, va="top", ha="right", fontweight="bold")
        y -= 0.11

    # Mini cross-tabulation table
    ax2.text(0.05, y - 0.03,
             "Cross-tabulation (rows=baseline, cols=alternative):",
             transform=ax2.transAxes, fontsize=FONT_BASE - 1.5, va="top", color="dimgrey")
    y -= 0.16

    short = {"underserved": "U", "below_average": "BA",
             "above_average": "AA", "well_served": "WS"}
    # Header row
    for j, alt_cat in enumerate(CAT_ORDER):
        ax2.text(0.22 + j * 0.19, y + 0.06, short[alt_cat],
                 transform=ax2.transAxes, fontsize=FONT_BASE - 2,
                 ha="center", fontweight="bold", color=CAT_COLORS[alt_cat])
    # Data rows
    for i, base_cat in enumerate(CAT_ORDER):
        ax2.text(0.04, y - i * 0.09, short[base_cat],
                 transform=ax2.transAxes, fontsize=FONT_BASE - 2,
                 va="top", fontweight="bold", color=CAT_COLORS[base_cat])
        for j, alt_cat in enumerate(CAT_ORDER):
            val     = ct.loc[base_cat, alt_cat]
            on_diag = (base_cat == alt_cat)
            ax2.text(0.22 + j * 0.19, y - i * 0.09, f"{val:,}",
                     transform=ax2.transAxes, fontsize=FONT_BASE - 2,
                     va="top", ha="center",
                     fontweight="bold" if on_diag else "normal",
                     color=CAT_COLORS[alt_cat] if on_diag else "dimgrey")

    fig.tight_layout()
    _save(fig, "05_category_comparison.png")
    return fig


# ---------------------------------------------------------------------------
# Chart 6 — Choropleth: districts coloured by baseline_index  (Q6)
# ---------------------------------------------------------------------------

def plot_choropleth(scores: pd.DataFrame = None,
                    geodf: gpd.GeoDataFrame = None) -> plt.Figure:
    """
    Answers Q6: What is the spatial pattern of emergency access across Peru?
    Are underserved districts geographically clustered or dispersed?

    Chart type — choropleth map via GeoPandas .plot():
    A choropleth is the natural choice for showing how a continuous index
    varies across administrative units while preserving spatial context
    (coastline, highlands, jungle regions).  An interactive Folium map was
    considered but cannot be saved as a static 300-DPI PNG.  A dot map was
    rejected because district centroids would obscure large Amazonian
    districts that are visually important to the analysis.

    Two panels: left uses a continuous colormap (plasma_r) to show the
    raw index value; right uses the four categorical colours from the rest
    of the report to show the access classification.
    """
    if scores is None:
        scores = _load_scores()
    if geodf is None:
        geodf = _load_geodataframe()

    # Merge baseline_index and category into the GeoDataFrame
    merge_cols = ["iddist", "baseline_index", "baseline_category"]
    score_sub  = scores[merge_cols].copy()
    score_sub["iddist"] = pd.to_numeric(score_sub["iddist"], errors="coerce")

    gdf = geodf.merge(score_sub, on="iddist", how="left")
    gdf["cat_color"] = gdf["baseline_category"].astype(str).map(CAT_COLORS).fillna("#cccccc")

    fig, axes = plt.subplots(1, 2, figsize=(18, 11))
    fig.suptitle(
        "Emergency Access Index — Peru District-Level Choropleth\n"
        "Baseline specification  |  2024-2026  |  1,873 districts",
        fontsize=FONT_BASE + 4, fontweight="bold", y=1.01,
    )

    # --- Left: continuous index ---
    ax = axes[0]
    gdf.plot(
        column="baseline_index",
        cmap="plasma_r",
        linewidth=0.05,
        edgecolor="white",
        legend=True,
        legend_kwds={
            "label":       "Baseline index",
            "orientation": "horizontal",
            "fraction":    0.03,
            "pad":         0.04,
            "shrink":      0.7,
        },
        missing_kwds={"color": "#cccccc", "label": "No data"},
        ax=ax,
    )
    ax.set_title("Continuous index value", fontsize=FONT_BASE + 1)
    ax.set_axis_off()

    # --- Right: categorical ---
    ax = axes[1]
    gdf.plot(
        color=gdf["cat_color"],
        linewidth=0.05,
        edgecolor="white",
        ax=ax,
    )
    ax.set_title("Access category (quartile thresholds)", fontsize=FONT_BASE + 1)
    ax.set_axis_off()
    ax.legend(
        handles=_cat_legend_handles() + [
            mpatches.Patch(color="#cccccc", label="No data")
        ],
        loc="lower left", frameon=True, fontsize=FONT_BASE - 1,
    )

    fig.tight_layout()
    _save(fig, "06_choropleth_baseline_index.png")
    return fig


# ---------------------------------------------------------------------------
# Run all charts
# ---------------------------------------------------------------------------

def plot_all() -> dict:
    """
    Generate all six charts in sequence, loading data once.

    Returns
    -------
    dict mapping chart name to matplotlib Figure.
    """
    print("\n[visualization] Loading data ...")
    scores = _load_scores()
    geodf  = _load_geodataframe()
    print(f"  Scores : {len(scores):,} districts x {len(scores.columns)} cols")
    print(f"  GeoDF  : {len(geodf):,} polygons  |  CRS: {geodf.crs}")

    charts = {}

    print("\n[1/6] Index distribution ...")
    charts["index_distribution"] = plot_index_distribution(scores)

    print("[2/6] Top-20 underserved ...")
    charts["top20_underserved"] = plot_top20_underserved(scores)

    print("[3/6] Scatter facility vs attendance ...")
    charts["scatter"] = plot_scatter_facility_vs_attendance(scores)

    print("[4/6] Boxplot distance by category ...")
    charts["boxplot"] = plot_boxplot_distance_by_category(scores)

    print("[5/6] Category comparison ...")
    charts["comparison"] = plot_category_comparison(scores)

    print("[6/6] Choropleth map ...")
    charts["choropleth"] = plot_choropleth(scores, geodf)

    print(f"\n[visualization] Done. All charts saved to: {FIGURES_DIR}")
    return charts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_all()
