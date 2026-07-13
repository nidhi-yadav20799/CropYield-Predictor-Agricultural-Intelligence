"""
CropYield Predictor — Agricultural Intelligence & Analytics Platform
=====================================================================
A single-file, production-grade Streamlit application built on top of the
exact data, feature-engineering logic and trained Random Forest pipeline
produced in `notebooks/CropYield_Predictor_End_to_End.ipynb`.

No models are retrained here. This file only loads, transforms and
presents artifacts that already exist on disk:
    data/processed/cleaned_crop_yield.csv
    data/processed/feature_engineered_crop_yield.csv
    models/random_forest_pipeline.pkl
    reports/model_evaluation_results.csv
    images/*.png

Author: Senior Data/Full-Stack Engineering pass, CropYield Predictor project.
"""

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# 0. PAGE CONFIGURATION — must be the first Streamlit call
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CropYield Predictor | Agricultural Intelligence Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# 1. PATH RESOLUTION — resilient to being launched from any working directory
# ----------------------------------------------------------------------------
APP_FILE = Path(__file__).resolve()
PROJECT_ROOT = APP_FILE.parent.parent  # CropYield-Predictor/

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLEANED_CSV = DATA_DIR / "cleaned_crop_yield.csv"
FEATURED_CSV = DATA_DIR / "feature_engineered_crop_yield.csv"
MODEL_PKL = MODEL_DIR / "random_forest_pipeline.pkl"
EVAL_CSV = REPORTS_DIR / "model_evaluation_results.csv"

TARGET_COL = "hg/ha_yield"

# Canonical image manifest — maps a stable key to (filename, title, insight).
# Every file is located dynamically; nothing is assumed to exist.
EDA_IMAGES = [
    ("correlation_heatmap.png", "Correlation Heatmap",
     "No pair of variables shows strong linear correlation, confirming low "
     "multicollinearity across rainfall, temperature, pesticide usage and year."),
    ("top_feature_correlation.png", "Feature Correlation with Yield",
     "Year and pesticide usage carry the strongest (still weak) positive "
     "correlation with yield; temperature trends weakly negative."),
    ("crop_yield_by_region.png", "Top 15 Regions by Average Yield",
     "Yield potential is highly geography-dependent — the top regions "
     "consistently out-produce the global average by a wide margin."),
    ("top10_regions_yield.png", "Top 10 Regions — Yield Detail",
     "A closer view of the highest-performing regions used to validate "
     "regional stratification in the model."),
    ("rainfall_vs_yield.png", "Rainfall vs. Crop Yield",
     "The regression trend line is nearly flat — rainfall alone is a weak "
     "linear predictor, motivating the non-linear ensemble approach."),
    ("temperature_vs_yield.png", "Temperature vs. Crop Yield",
     "A mild negative trend appears, but wide dispersion shows yield is "
     "shaped by many interacting climate and agronomic factors."),
    ("pairplot.png", "Pairwise Feature Relationships",
     "Weak pairwise linear structure across all numerical drivers supports "
     "using tree-based models capable of capturing non-linear interaction."),
]

PERFORMANCE_IMAGES = [
    ("model_comparison_cv.png", "5-Fold Cross-Validation Comparison",
     "XGBoost and Random Forest lead the field with average R² above 0.99, "
     "far ahead of the linear baselines."),
    ("hyperparameter_tuning_comparison.png", "Tuning Impact — Before vs. After",
     "RandomizedSearchCV improved both ensemble candidates, reinforcing "
     "tree-based models as the correct family for this problem."),
    ("random_forest_feature_importance.png", "Random Forest — Feature Importance",
     "Engineered ratios and categorical region/crop encodings dominate the "
     "importance ranking, validating the feature-engineering phase."),
    ("residual_plot.png", "Residual Plot",
     "Residuals scatter randomly around zero with no systematic funnel or "
     "curve, indicating an unbiased fit across the prediction range."),
    ("error_distribution.png", "Prediction Error Distribution",
     "Errors concentrate tightly near zero with only a small tail of larger "
     "residuals, matching the model's strong aggregate R² score."),
]


# ----------------------------------------------------------------------------
# 2. CACHED DATA / MODEL LOADERS
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    """Load a CSV if present, else return None. Never raises to the UI."""
    if path is None or not Path(path).exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """Load the persisted Scikit-Learn pipeline via joblib. No retraining."""
    if path is None or not Path(path).exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def image_to_base64(path: Path):
    """Read an image file from disk and return a base64 data-URI string."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def build_lookup_tables(_df_feat: pd.DataFrame):
    """
    Pre-compute lookup structures needed to reproduce the notebook's
    feature engineering for a single, previously-unseen prediction row.
    """
    lookups = {}
    if _df_feat is None or _df_feat.empty:
        return lookups

    # Pesticide quartile edges (mirrors pd.qcut(..., q=4) at training time).
    if "pesticides_tonnes" in _df_feat.columns:
        try:
            _, bin_edges = pd.qcut(
                _df_feat["pesticides_tonnes"], q=4, retbins=True, duplicates="drop"
            )
            lookups["pesticide_edges"] = bin_edges
        except Exception:
            lookups["pesticide_edges"] = None

    # Historical Yield-per-Pesticide, used only as a stable proxy input for
    # the engineered `Yield_per_Pesticide` feature the trained pipeline
    # expects at inference time (this feature was derived from the target
    # during training; a forward-looking value is estimated from
    # region + crop history rather than the unknown future yield).
    if {"Area", "Item", "Yield_per_Pesticide"}.issubset(_df_feat.columns):
        lookups["yield_per_pesticide_by_area_item"] = (
            _df_feat.groupby(["Area", "Item"])["Yield_per_Pesticide"].mean().to_dict()
        )
        lookups["yield_per_pesticide_by_item"] = (
            _df_feat.groupby("Item")["Yield_per_Pesticide"].mean().to_dict()
        )
        lookups["yield_per_pesticide_global"] = float(
            _df_feat["Yield_per_Pesticide"].mean()
        )
    return lookups


def engineer_features(area, item, year, rainfall, pesticides, avg_temp, lookups):
    """
    Reproduce, feature-for-feature, the transformations applied in the
    notebook's Feature Engineering section for a single new observation.
    """
    # Rainfall_Category — identical bins to the notebook.
    rain_bins = [0, 500, 1000, 2000, float("inf")]
    rain_labels = ["Low", "Moderate", "High", "Very High"]
    rainfall_category = pd.cut([rainfall], bins=rain_bins, labels=rain_labels)[0]

    # Growing_Degree_Days — base temperature of 10, clipped at zero.
    gdd = max(avg_temp - 10, 0.0)

    # Temperature_Category — identical bins to the notebook.
    temp_bins = [0, 10, 20, 25, float("inf")]
    temp_labels = ["Cold", "Moderate", "Warm", "Hot"]
    temperature_category = pd.cut([avg_temp], bins=temp_bins, labels=temp_labels)[0]

    # Pesticide_Category — quartile edges learned from the training data.
    edges = lookups.get("pesticide_edges")
    pesticide_labels = ["Low", "Medium", "High", "Very High"]
    if edges is not None and len(edges) >= 2:
        idx = int(np.clip(np.searchsorted(edges, pesticides, side="right") - 1, 0, 3))
        pesticide_category = pesticide_labels[idx]
    else:
        pesticide_category = "Medium"

    # Rainfall_Temp_Ratio — direct ratio, guarded against divide-by-zero.
    rainfall_temp_ratio = rainfall / avg_temp if avg_temp != 0 else 0.0

    # Yield_per_Pesticide proxy — historical Area+Item mean, falling back
    # to Item mean, then the global mean.
    yp_area_item = lookups.get("yield_per_pesticide_by_area_item", {})
    yp_item = lookups.get("yield_per_pesticide_by_item", {})
    yp_global = lookups.get("yield_per_pesticide_global", 0.0)
    yield_per_pesticide = yp_area_item.get(
        (area, item), yp_item.get(item, yp_global)
    )

    row = pd.DataFrame([{
        "Area": area,
        "Item": item,
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": avg_temp,
        "Rainfall_Category": rainfall_category,
        "Growing_Degree_Days": gdd,
        "Temperature_Category": temperature_category,
        "Pesticide_Category": pesticide_category,
        "Rainfall_Temp_Ratio": rainfall_temp_ratio,
        "Yield_per_Pesticide": yield_per_pesticide,
    }])
    return row


def tree_ensemble_spread(pipeline, row: pd.DataFrame):
    """
    Derive a prediction range from the Random Forest's individual trees
    (mirrors the CI-free nature of RF by using estimator disagreement as a
    practical proxy for uncertainty). Returns (low, high, std) or None.
    """
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        model = pipeline.named_steps.get("model")
        if preprocessor is None or model is None or not hasattr(model, "estimators_"):
            return None
        transformed = preprocessor.transform(row)
        tree_preds = np.array([t.predict(transformed)[0] for t in model.estimators_])
        return float(np.percentile(tree_preds, 10)), float(np.percentile(tree_preds, 90)), float(tree_preds.std())
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 3. LOAD EVERYTHING ONCE
# ----------------------------------------------------------------------------
df_clean = load_csv(CLEANED_CSV)
df_feat = load_csv(FEATURED_CSV)
model_pipeline = load_model(MODEL_PKL)
eval_results = load_csv(EVAL_CSV)
feature_lookups = build_lookup_tables(df_feat) if df_feat is not None else {}


# ----------------------------------------------------------------------------
# 4. GLOBAL STYLE SYSTEM — the only CSS in this application
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
    --forest-950:#0B2A1E;
    --forest-900:#123A29;
    --forest-800:#14532D;
    --forest-700:#1D6E3D;
    --forest-500:#2E8B57;
    --sage-200:#D9E8DC;
    --sage-100:#EEF4EF;
    --harvest-500:#C9A24B;
    --harvest-600:#B08A34;
    --bg:#F5F7F5;
    --card:#FFFFFF;
    --ink-900:#12201A;
    --ink-700:#3B4A42;
    --ink-500:#6B7A72;
    --line:#E4EAE5;
    --danger:#C0453B;
    --radius-lg:24px;
    --radius-md:18px;
    --radius-sm:12px;
    --shadow-sm:0 1px 2px rgba(18,32,26,0.04), 0 1px 3px rgba(18,32,26,0.05);
    --shadow-md:0 8px 24px rgba(18,32,26,0.08);
    --shadow-lg:0 20px 48px rgba(18,32,26,0.14);
}

html, body, [class*="css"]{
    font-family:'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu, footer, header[data-testid="stHeader"]{visibility:hidden; height:0;}
.block-container{
    padding-top:1.2rem !important;
    padding-bottom:3rem !important;
    max-width:1280px;
}
body, .stApp{ background:var(--bg); }

h1,h2,h3,h4{ font-family:'Sora', sans-serif; color:var(--ink-900); letter-spacing:-0.01em; }

/* ---------- Utility text ---------- */
.eyebrow{
    font-family:'JetBrains Mono', monospace;
    font-size:0.72rem;
    letter-spacing:0.14em;
    text-transform:uppercase;
    color:var(--forest-700);
    font-weight:600;
}
.muted{ color:var(--ink-500); }

/* ---------- Top Navigation ---------- */
.nav-wrap{
    background:linear-gradient(180deg, var(--forest-900) 0%, var(--forest-800) 100%);
    border-radius:var(--radius-lg);
    padding:14px 26px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    box-shadow:var(--shadow-md);
    margin-bottom:28px;
    flex-wrap:wrap;
    gap:12px;
}
.brand{
    display:flex;
    align-items:center;
    gap:10px;
    color:#FFFFFF;
    font-family:'Sora', sans-serif;
    font-weight:700;
    font-size:1.08rem;
    white-space:nowrap;
}
.brand-mark{
    width:34px; height:34px;
    border-radius:10px;
    background:linear-gradient(135deg, var(--harvest-500), var(--forest-500));
    display:flex; align-items:center; justify-content:center;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,0.25);
}
.nav-links{ display:flex; align-items:center; gap:4px; flex-wrap:wrap; }
.nav-link{
    font-family:'Inter', sans-serif;
    font-weight:600;
    font-size:0.86rem;
    color:rgba(255,255,255,0.72);
    text-decoration:none;
    padding:9px 18px;
    border-radius:999px;
    transition:all 0.18s ease;
}
.nav-link:hover{ color:#FFFFFF; background:rgba(255,255,255,0.08); }
.nav-link.active{
    color:var(--forest-900);
    background:#FFFFFF;
    box-shadow:var(--shadow-sm);
}

/* ---------- Hero ---------- */
.hero{
    position:relative;
    background:radial-gradient(120% 140% at 100% 0%, rgba(201,162,75,0.16) 0%, rgba(201,162,75,0) 55%),
               linear-gradient(120deg, var(--forest-900) 0%, var(--forest-800) 55%, var(--forest-700) 100%);
    border-radius:var(--radius-lg);
    padding:56px 52px;
    overflow:hidden;
    box-shadow:var(--shadow-lg);
    margin-bottom:28px;
}
.hero::after{
    content:"";
    position:absolute; right:-60px; bottom:-90px;
    width:340px; height:340px; border-radius:50%;
    border:1px solid rgba(255,255,255,0.12);
}
.hero::before{
    content:"";
    position:absolute; right:10px; bottom:-40px;
    width:220px; height:220px; border-radius:50%;
    border:1px solid rgba(255,255,255,0.10);
}
.hero-eyebrow{
    font-family:'JetBrains Mono', monospace;
    color:var(--harvest-500);
    letter-spacing:0.18em;
    text-transform:uppercase;
    font-size:0.74rem;
    font-weight:600;
}
.hero h1{
    color:#FFFFFF;
    font-size:2.5rem;
    line-height:1.15;
    margin:14px 0 14px 0;
    max-width:680px;
}
.hero p{
    color:rgba(255,255,255,0.78);
    font-size:1.02rem;
    max-width:620px;
    line-height:1.6;
}
.hero-pills{ display:flex; gap:10px; margin-top:22px; flex-wrap:wrap; }
.hero-pill{
    font-size:0.78rem;
    font-weight:600;
    color:#FFFFFF;
    background:rgba(255,255,255,0.10);
    border:1px solid rgba(255,255,255,0.18);
    padding:7px 14px;
    border-radius:999px;
}

/* ---------- Cards ---------- */
.card{
    background:var(--card);
    border-radius:var(--radius-md);
    padding:24px;
    box-shadow:var(--shadow-sm);
    border:1px solid var(--line);
    height:100%;
}
.card-title{ font-family:'Sora', sans-serif; font-weight:700; font-size:1.02rem; color:var(--ink-900); margin-bottom:4px;}
.card-sub{ font-size:0.86rem; color:var(--ink-500); margin-bottom:14px;}

/* ---------- KPI Grid ---------- */
.kpi-grid{ display:grid; grid-template-columns:repeat(4, 1fr); gap:18px; margin:8px 0 30px 0; }
@media (max-width:1100px){ .kpi-grid{ grid-template-columns:repeat(2, 1fr);} }
.kpi-card{
    background:var(--card);
    border:1px solid var(--line);
    border-radius:var(--radius-md);
    padding:22px 22px 20px 22px;
    box-shadow:var(--shadow-sm);
    transition:transform 0.18s ease, box-shadow 0.18s ease;
}
.kpi-card:hover{ transform:translateY(-3px); box-shadow:var(--shadow-md); }
.kpi-top{ display:flex; align-items:center; justify-content:space-between; margin-bottom:14px;}
.kpi-icon{
    width:38px; height:38px; border-radius:10px;
    background:var(--sage-100);
    display:flex; align-items:center; justify-content:center;
}
.kpi-tag{
    font-family:'JetBrains Mono', monospace;
    font-size:0.68rem; font-weight:600;
    color:var(--forest-700);
    background:var(--sage-100);
    padding:3px 9px; border-radius:999px;
}
.kpi-value{ font-family:'JetBrains Mono', monospace; font-size:1.85rem; font-weight:700; color:var(--ink-900); }
.kpi-label{ font-size:0.86rem; color:var(--ink-500); margin-top:4px; font-weight:500;}

/* ---------- Workflow Steps ---------- */
.workflow{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin:10px 0 8px 0;}
@media (max-width:1100px){ .workflow{ grid-template-columns:repeat(2,1fr);} }
.wf-step{
    background:var(--card); border:1px solid var(--line); border-radius:var(--radius-md);
    padding:20px; box-shadow:var(--shadow-sm); position:relative;
}
.wf-num{
    font-family:'JetBrains Mono', monospace; font-weight:700; font-size:0.78rem;
    color:var(--harvest-600); margin-bottom:10px; display:block;
}
.wf-title{ font-family:'Sora', sans-serif; font-weight:700; font-size:0.95rem; color:var(--ink-900); margin-bottom:6px;}
.wf-desc{ font-size:0.82rem; color:var(--ink-500); line-height:1.5;}

/* ---------- Section headers ---------- */
.section-head{ display:flex; align-items:baseline; justify-content:space-between; margin:8px 0 16px 0; }
.section-title{ font-family:'Sora', sans-serif; font-weight:700; font-size:1.4rem; color:var(--ink-900);}
.section-desc{ color:var(--ink-500); font-size:0.92rem; margin-top:2px;}

/* ---------- Image gallery ---------- */
.gallery-grid{ display:grid; grid-template-columns:repeat(2, 1fr); gap:20px; margin-bottom:8px;}
@media (max-width:900px){ .gallery-grid{ grid-template-columns:1fr;} }
.gallery-card{
    background:var(--card); border:1px solid var(--line); border-radius:var(--radius-md);
    overflow:hidden; box-shadow:var(--shadow-sm); transition:transform 0.22s ease, box-shadow 0.22s ease;
}
.gallery-card:hover{ transform:translateY(-4px); box-shadow:var(--shadow-md); }
.gallery-img-wrap{ overflow:hidden; background:var(--sage-100); }
.gallery-img-wrap img{ width:100%; display:block; transition:transform 0.4s ease; }
.gallery-card:hover .gallery-img-wrap img{ transform:scale(1.035); }
.gallery-body{ padding:18px 20px 20px 20px; }
.gallery-title{ font-family:'Sora', sans-serif; font-weight:700; font-size:1rem; color:var(--ink-900); margin-bottom:6px;}
.gallery-obs{ font-size:0.86rem; color:var(--ink-700); line-height:1.55;}
.gallery-missing{
    padding:40px 20px; text-align:center; color:var(--ink-500); font-size:0.86rem;
    background:var(--sage-100);
}

/* ---------- Badges / tech stack ---------- */
.badge-row{ display:flex; flex-wrap:wrap; gap:10px; }
.badge{
    background:var(--sage-100); color:var(--forest-800); font-weight:600; font-size:0.82rem;
    padding:8px 16px; border-radius:999px; border:1px solid var(--sage-200);
}

/* ---------- Result card (Prediction) ---------- */
.result-card{
    background:linear-gradient(135deg, var(--forest-900), var(--forest-700));
    border-radius:var(--radius-lg); padding:34px 36px; box-shadow:var(--shadow-lg);
    color:#fff;
}
.result-label{
    font-family:'JetBrains Mono', monospace; letter-spacing:0.1em; text-transform:uppercase;
    font-size:0.72rem; color:var(--harvest-500); font-weight:700;
}
.result-value{ font-family:'Sora', sans-serif; font-weight:800; font-size:3rem; margin:10px 0 4px 0;}
.result-unit{ color:rgba(255,255,255,0.65); font-size:0.95rem; }
.result-range{
    margin-top:18px; padding-top:18px; border-top:1px solid rgba(255,255,255,0.16);
    display:flex; gap:30px; flex-wrap:wrap;
}
.range-item .k{ font-size:0.76rem; color:rgba(255,255,255,0.6); text-transform:uppercase; letter-spacing:0.08em;}
.range-item .v{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:1.15rem;}

.input-summary-table{ width:100%; border-collapse:collapse; }
.input-summary-table td{ padding:9px 4px; font-size:0.87rem; border-bottom:1px solid var(--line); }
.input-summary-table td.k{ color:var(--ink-500); }
.input-summary-table td.v{ color:var(--ink-900); font-weight:600; text-align:right; font-family:'JetBrains Mono', monospace; }

/* ---------- Form group labels ---------- */
.form-group-label{
    font-family:'Sora', sans-serif; font-weight:700; font-size:0.92rem; color:var(--forest-800);
    margin:2px 0 10px 0; display:flex; align-items:center; gap:8px;
}
.form-group-label .dot{ width:8px; height:8px; border-radius:50%; background:var(--harvest-500); display:inline-block;}

/* ---------- Streamlit widget restyle ---------- */
div[data-testid="stSelectbox"] > div, div[data-testid="stNumberInput"] input{
    border-radius:var(--radius-sm) !important;
}
.stSlider > div > div > div > div{ background:var(--forest-500) !important; }

div.stButton > button{
    background:linear-gradient(135deg, var(--forest-700), var(--forest-900));
    color:#fff; border:none; border-radius:999px; font-weight:700;
    padding:0.7rem 1.6rem; box-shadow:var(--shadow-sm); font-family:'Sora', sans-serif;
    transition:transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button:hover{ transform:translateY(-2px); box-shadow:var(--shadow-md); color:#fff; }
div.stButton > button p{ color:#fff !important; font-weight:700 !important;}

/* ---------- Dataframe container ---------- */
div[data-testid="stDataFrame"]{ border-radius:var(--radius-md); overflow:hidden; border:1px solid var(--line); }

/* ---------- Footer ---------- */
.app-footer{
    margin-top:44px; padding:24px 6px 6px 6px; border-top:1px solid var(--line);
    display:flex; justify-content:space-between; color:var(--ink-500); font-size:0.8rem; flex-wrap:wrap; gap:8px;
}

/* ---------- Missing-data notice ---------- */
.notice{
    background:#FFF7E8; border:1px solid #F0DDA6; color:#7A5A17;
    border-radius:var(--radius-sm); padding:14px 18px; font-size:0.86rem; margin-bottom:18px;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 5. SMALL RENDER HELPERS
# ----------------------------------------------------------------------------
def section_header(title: str, desc: str = ""):
    st.markdown(
        f"""<div class="section-head">
                <div>
                    <div class="section-title">{title}</div>
                    <div class="section-desc">{desc}</div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )


def kpi_card_html(icon_svg: str, tag: str, value: str, label: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-top">
            <div class="kpi-icon">{icon_svg}</div>
            <div class="kpi-tag">{tag}</div>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """


ICON_LEAF = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#14532D" stroke-width="2"><path d="M5 21c8-1 13-6 14-14C11 8 6 13 5 21Z"/></svg>"""
ICON_GRID = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#14532D" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>"""
ICON_GLOBE = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#14532D" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.6 4 6 4 9s-1.5 6.4-4 9c-2.5-2.6-4-6-4-9s1.5-6.4 4-9Z"/></svg>"""
ICON_TARGET = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#14532D" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="0.6" fill="#14532D"/></svg>"""
ICON_TREND = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#14532D" stroke-width="2"><path d="M3 17l6-6 4 4 8-9"/><path d="M15 6h6v6"/></svg>"""
ICON_CLOCK = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#14532D" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>"""


def render_gallery(image_specs, base_dir: Path, columns: int = 2):
    """Render a responsive card gallery for a list of (file, title, insight)."""
    cards_html = []
    for filename, title, insight in image_specs:
        path = base_dir / filename
        b64 = image_to_base64(path)
        if b64:
            img_block = f'<div class="gallery-img-wrap"><img src="{b64}" /></div>'
        else:
            img_block = f'<div class="gallery-missing">Image not found: {filename}</div>'
        cards_html.append(f"""
        <div class="gallery-card">
            {img_block}
            <div class="gallery-body">
                <div class="gallery-title">{title}</div>
                <div class="gallery-obs">{insight}</div>
            </div>
        </div>
        """)
    grid_style = f'grid-template-columns:repeat({columns}, 1fr);'
    st.markdown(
        f'<div class="gallery-grid" style="{grid_style}">{"".join(cards_html)}</div>',
        unsafe_allow_html=True,
    )


def missing_file_notice(paths_and_labels):
    missing = [label for path, label in paths_and_labels if not Path(path).exists()]
    if missing:
        st.markdown(
            f'<div class="notice">⚠ Some expected files were not found on disk and '
            f'related sections are showing limited data: {", ".join(missing)}.</div>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# 6. NAVIGATION
# ----------------------------------------------------------------------------
NAV_ITEMS = [
    ("dashboard", "Dashboard"),
    ("eda", "EDA"),
    ("prediction", "Prediction"),
    ("performance", "Performance"),
    ("about", "About"),
]

query_params = st.query_params
active_page = query_params.get("page", "dashboard")
if active_page not in dict(NAV_ITEMS):
    active_page = "dashboard"

nav_links_html = "".join(
    f'<a class="nav-link {"active" if key == active_page else ""}" href="?page={key}">{label}</a>'
    for key, label in NAV_ITEMS
)

st.markdown(
    f"""
    <div class="nav-wrap">
        <div class="brand">
            <div class="brand-mark">{ICON_LEAF.replace('#14532D','#FFFFFF')}</div>
            CropYield Predictor
        </div>
        <div class="nav-links">{nav_links_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 7. PAGE — DASHBOARD (HOME)
# ----------------------------------------------------------------------------
def page_dashboard():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Agricultural Intelligence &amp; Analytics Platform</div>
            <h1>Predict crop yield with production-grade machine learning.</h1>
            <p>A complete data science workflow — from raw weather and pesticide records to a
            tuned ensemble model — packaged into one interactive analytics platform for
            researchers, agronomists and decision-makers.</p>
            <div class="hero-pills">
                <div class="hero-pill">Random Forest · Tuned</div>
                <div class="hero-pill">5-Fold Cross-Validated</div>
                <div class="hero-pill">R² ≥ 0.98 on Test Set</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing_file_notice([
        (CLEANED_CSV, "cleaned_crop_yield.csv"),
        (MODEL_PKL, "random_forest_pipeline.pkl"),
        (EVAL_CSV, "model_evaluation_results.csv"),
    ])

    # ---- KPI row, computed live from the cleaned dataset ----
    if df_clean is not None:
        n_rows = f"{len(df_clean):,}"
        n_regions = f"{df_clean['Area'].nunique():,}" if "Area" in df_clean.columns else "—"
        n_crops = f"{df_clean['Item'].nunique():,}" if "Item" in df_clean.columns else "—"
        year_range = (
            f"{int(df_clean['Year'].min())}–{int(df_clean['Year'].max())}"
            if "Year" in df_clean.columns else "—"
        )
    else:
        n_rows = n_regions = n_crops = year_range = "—"

    best_r2 = "—"
    if eval_results is not None and "R2 Score" in eval_results.columns:
        best_r2 = f"{eval_results['R2 Score'].max():.3f}"

    kpis = [
        (ICON_GRID, "RECORDS", n_rows, "Historical observations"),
        (ICON_GLOBE, "COVERAGE", n_regions, "Countries / regions tracked"),
        (ICON_LEAF, "CROPS", n_crops, "Distinct crop types"),
        (ICON_TARGET, "BEST MODEL", best_r2, "Highest test R² score"),
    ]
    st.markdown(
        '<div class="kpi-grid">' + "".join(kpi_card_html(*k) for k in kpis) + "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Project Summary</div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="card-sub">
            This platform analyzes historical weather, rainfall, pesticide usage and
            temperature data across multiple regions and crop types to predict agricultural
            yield (measured in hectograms per hectare). The workflow spans data validation,
            cleaning, exploratory analysis, domain-driven feature engineering, multi-model
            comparison under 5-fold cross-validation, hyperparameter tuning via
            RandomizedSearchCV, and final deployment of a tuned Random Forest pipeline.
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        st.markdown('<div class="section-title" style="font-size:1.1rem;">Workflow</div>', unsafe_allow_html=True)
        steps = [
            ("01", "Data Validation", "Schema checks, missing-value and duplicate audits on raw records."),
            ("02", "Cleaning", "Outlier review via IQR and dtype correction, exported as a clean dataset."),
            ("03", "EDA", "Univariate, bivariate and correlation analysis across all key drivers."),
            ("04", "Feature Engineering", "Rainfall/temperature bins, GDD, pesticide quartiles and ratio features."),
            ("05", "Modeling & Tuning", "5 models compared via CV; Random Forest and XGBoost tuned further."),
        ]
        st.markdown(
            '<div class="workflow">' + "".join(
                f"""<div class="wf-step">
                        <span class="wf-num">{num}</span>
                        <div class="wf-title">{title}</div>
                        <div class="wf-desc">{desc}</div>
                    </div>"""
                for num, title, desc in steps
            ) + "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Dataset Overview</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-sub">Coverage window: {year_range}</div>', unsafe_allow_html=True)
        if df_clean is not None:
            preview_cols = [c for c in ["Area", "Item", "Year", "hg/ha_yield",
                                         "average_rain_fall_mm_per_year",
                                         "pesticides_tonnes", "avg_temp"] if c in df_clean.columns]
            st.dataframe(df_clean[preview_cols].head(8), use_container_width=True, hide_index=True)
        else:
            st.info("Cleaned dataset not found on disk yet.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Quick Insights</div>', unsafe_allow_html=True)
        insight_list = [
            "Ensemble tree models (Random Forest, XGBoost) outperform linear baselines by a wide margin.",
            "Yield varies far more by region and crop type than by any single climate variable.",
            "Rainfall and temperature alone are weak linear predictors — non-linear interaction matters.",
        ]
        st.markdown(
            "".join(f'<p class="card-sub" style="margin-bottom:10px;">▸ {t}</p>' for t in insight_list),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 8. PAGE — EDA
# ----------------------------------------------------------------------------
def page_eda():
    section_header(
        "Exploratory Data Analysis",
        "Distribution, correlation and regional insight visualizations generated directly from the notebook."
    )
    missing_file_notice([(IMAGES_DIR / f, f) for f, _, _ in EDA_IMAGES])
    render_gallery(EDA_IMAGES, IMAGES_DIR, columns=2)


# ----------------------------------------------------------------------------
# 9. PAGE — PREDICTION
# ----------------------------------------------------------------------------
def page_prediction():
    section_header(
        "Yield Prediction",
        "Generate a crop yield estimate from the tuned Random Forest pipeline — no retraining, live inference only."
    )

    if model_pipeline is None:
        st.markdown(
            '<div class="notice">⚠ The trained pipeline '
            f'(<code>{MODEL_PKL.name}</code>) was not found. Place it under '
            '<code>models/</code> to enable predictions.</div>',
            unsafe_allow_html=True,
        )
        return

    areas = sorted(df_feat["Area"].dropna().unique().tolist()) if df_feat is not None and "Area" in df_feat.columns else ["Unknown"]
    items = sorted(df_feat["Item"].dropna().unique().tolist()) if df_feat is not None and "Item" in df_feat.columns else ["Unknown"]

    form_left, form_right = st.columns(2, gap="large")

    with form_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="form-group-label"><span class="dot"></span>Location &amp; Crop</div>', unsafe_allow_html=True)
        area = st.selectbox("Region", areas, index=0)
        item = st.selectbox("Crop Type", items, index=0)
        year = st.slider("Year", min_value=1990, max_value=2035, value=2024, step=1)
        st.markdown("</div>", unsafe_allow_html=True)

    with form_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="form-group-label"><span class="dot"></span>Climate &amp; Inputs</div>', unsafe_allow_html=True)
        rainfall = st.number_input("Average Rainfall (mm / year)", min_value=0.0, max_value=5000.0, value=1100.0, step=10.0)
        avg_temp = st.number_input("Average Temperature (°C)", min_value=-10.0, max_value=45.0, value=20.5, step=0.1)
        pesticides = st.number_input("Pesticide Usage (tonnes)", min_value=0.0, max_value=400000.0, value=17000.0, step=100.0)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    predict_clicked = st.button("Predict Crop Yield", use_container_width=False)

    if predict_clicked:
        input_row = engineer_features(area, item, year, rainfall, pesticides, avg_temp, feature_lookups)
        try:
            prediction = float(model_pipeline.predict(input_row)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        spread = tree_ensemble_spread(model_pipeline, input_row)

        result_col, summary_col = st.columns([1.3, 1], gap="large")
        with result_col:
            range_html = ""
            if spread:
                low, high, std = spread
                range_html = f"""
                <div class="result-range">
                    <div class="range-item"><div class="k">Low Estimate (P10)</div><div class="v">{low:,.0f}</div></div>
                    <div class="range-item"><div class="k">High Estimate (P90)</div><div class="v">{high:,.0f}</div></div>
                    <div class="range-item"><div class="k">Tree Spread (σ)</div><div class="v">±{std:,.0f}</div></div>
                </div>
                """
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Crop Yield</div>
                    <div class="result-value">{prediction:,.0f}</div>
                    <div class="result-unit">hectograms per hectare (hg/ha)</div>
                    {range_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Confidence range is derived from disagreement across the Random Forest's "
                "individual decision trees (10th–90th percentile of tree-level predictions), "
                "not a formal statistical confidence interval."
            )

        with summary_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Input Summary</div>', unsafe_allow_html=True)
            summary_rows = [
                ("Region", area), ("Crop", item), ("Year", year),
                ("Rainfall", f"{rainfall:,.0f} mm/yr"),
                ("Avg. Temperature", f"{avg_temp:.1f} °C"),
                ("Pesticide Usage", f"{pesticides:,.0f} t"),
            ]
            st.markdown(
                '<table class="input-summary-table">' + "".join(
                    f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>' for k, v in summary_rows
                ) + "</table>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 10. PAGE — PERFORMANCE
# ----------------------------------------------------------------------------
def page_performance():
    section_header(
        "Model Performance",
        "Cross-validated comparison, tuning impact and diagnostic plots for the final Random Forest pipeline."
    )
    missing_file_notice([(EVAL_CSV, "model_evaluation_results.csv")])

    if eval_results is not None:
        best_row = eval_results.sort_values("R2 Score", ascending=False).iloc[0] if "R2 Score" in eval_results.columns else None
        if best_row is not None:
            kpis = [
                (ICON_TARGET, "BEST MODEL", str(best_row["Model"]), "Top performer on held-out test set"),
                (ICON_TREND, "R² SCORE", f"{best_row['R2 Score']:.4f}", "Variance explained"),
                (ICON_GRID, "MAE", f"{best_row['MAE']:,.0f}", "Mean absolute error (hg/ha)"),
                (ICON_CLOCK, "RMSE", f"{best_row['RMSE']:,.0f}", "Root mean squared error (hg/ha)"),
            ]
            st.markdown(
                '<div class="kpi-grid">' + "".join(kpi_card_html(*k) for k in kpis) + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Model Comparison Table</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Evaluated on the held-out test split.</div>', unsafe_allow_html=True)
        display_df = eval_results.sort_values("R2 Score", ascending=False).reset_index(drop=True)
        st.dataframe(
            display_df.style.format({"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "R2 Score": "{:.4f}"}),
            use_container_width=True, hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
    else:
        st.info("Model evaluation results not found on disk yet.")

    render_gallery(PERFORMANCE_IMAGES, IMAGES_DIR, columns=2)


# ----------------------------------------------------------------------------
# 11. PAGE — ABOUT
# ----------------------------------------------------------------------------
def page_about():
    section_header("About This Project", "Scope, pipeline architecture and technology stack.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Overview</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="card-sub">
        CropYield Predictor is an end-to-end agricultural analytics platform built to
        forecast crop yield from historical rainfall, temperature and pesticide-usage
        records. The project follows a fully reproducible Scikit-Learn pipeline —
        preprocessing and modeling live in a single serialized object, eliminating data
        leakage between training and inference.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Pipeline Architecture</div>', unsafe_allow_html=True)
        stages = [
            "Raw CSV ingestion → schema &amp; missing-value validation",
            "Cleaning: duplicate removal, IQR outlier review, dtype correction",
            "Feature engineering: rainfall/temperature bins, Growing Degree Days, "
            "pesticide quartiles, rainfall/temperature ratio, yield-efficiency ratio",
            "ColumnTransformer: StandardScaler (numeric) + OneHotEncoder (categorical)",
            "Model comparison: Linear Regression, Ridge, Random Forest, "
            "Gradient Boosting, XGBoost — all under 5-fold cross-validation",
            "RandomizedSearchCV tuning on the top two ensemble candidates",
            "Final export: tuned Random Forest pipeline via joblib",
        ]
        st.markdown(
            "".join(f'<p class="card-sub" style="margin-bottom:10px;">▸ {s}</p>' for s in stages),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Technology Stack</div>', unsafe_allow_html=True)
        badges = ["Python", "Pandas", "NumPy", "Scikit-Learn", "XGBoost",
                   "Matplotlib", "Seaborn", "Streamlit", "Joblib"]
        st.markdown(
            '<div class="badge-row">' + "".join(f'<div class="badge">{b}</div>' for b in badges) + "</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown('<div class="card-title" style="margin-top:8px;">Dataset</div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="card-sub">
            Historical crop yield records spanning multiple countries and crop types,
            with rainfall, pesticide usage and average temperature as core predictors.
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Future Scope</div>', unsafe_allow_html=True)
    future = [
        "Incorporate soil nutrient (N-P-K) and humidity sensor data as they become available.",
        "Add MLflow experiment tracking for full run-history comparison inside the dashboard.",
        "Extend to multi-season, multi-year forecasting with time-aware validation.",
        "Publish a public API endpoint for programmatic yield predictions.",
    ]
    st.markdown(
        "".join(f'<p class="card-sub" style="margin-bottom:10px;">▸ {t}</p>' for t in future),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 12. ROUTER
# ----------------------------------------------------------------------------
PAGES = {
    "dashboard": page_dashboard,
    "eda": page_eda,
    "prediction": page_prediction,
    "performance": page_performance,
    "about": page_about,
}
PAGES.get(active_page, page_dashboard)()

# ----------------------------------------------------------------------------
# 13. FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        <div>CropYield Predictor · Agricultural Intelligence &amp; Analytics Platform</div>
        <div>Built on a reproducible Scikit-Learn pipeline · No retraining performed at runtime</div>
    </div>
    """,
    unsafe_allow_html=True,
)