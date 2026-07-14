"""
CropYield Predictor — Agricultural Intelligence & Analytics Platform
=====================================================================
A single-file, production-grade Streamlit application built on top of the
exact data, feature-engineering logic and trained Random Forest pipeline
produced in `notebooks/CropYield_Predictor_End_to_End.ipynb`.

Nothing about the data pipeline, model loading or prediction logic is
altered in this revision — only the interface layer has been rebuilt as a
premium SaaS-grade experience (glass morphism, motion, Plotly charts).

No models are retrained here. This file only loads, transforms and
presents artifacts that already exist on disk:
    data/processed/cleaned_crop_yield.csv
    data/processed/feature_engineered_crop_yield.csv
    models/random_forest_pipeline.pkl
    reports/model_evaluation_results.csv
    images/*.png
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
#    (UNCHANGED — dataset / model locations are not modified in this pass)
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


# ----------------------------------------------------------------------------
# 1b. SAFE HTML RENDERER
# ----------------------------------------------------------------------------
def html(fragment: str) -> None:
    """
    Render an HTML/CSS fragment through st.markdown safely.

    Streamlit's Markdown parser follows CommonMark: any line that opens with
    4+ leading spaces immediately after a blank line is treated as an
    *indented code block* and printed as literal text instead of being
    rendered as HTML. Multi-line HTML strings written with Python's natural
    indentation (matching the surrounding function body) trip this rule
    constantly — that is what caused raw `<div>` / CSS text to appear on
    screen in the previous build.

    This helper strips leading whitespace from every line before handing
    the fragment to Streamlit, so no HTML string can ever be mistaken for a
    code block, regardless of how it is indented in the source file.
    """
    flush_left = "\n".join(line.lstrip() for line in fragment.split("\n"))
    st.markdown(flush_left, unsafe_allow_html=True)

# Canonical image manifest — maps a stable key to (filename, title, insight).
# Each entry: (filename, title, technical observation, business insight)
EDA_IMAGES = [
    ("correlation_heatmap.png", "Correlation Heatmap",
     "No pair of variables shows strong linear correlation, confirming low "
     "multicollinearity across rainfall, temperature, pesticide usage and year.",
     "Safe to feed all four raw drivers into the model without redundancy risk "
     "or unstable coefficients from collinear inputs."),
    ("top_feature_correlation.png", "Feature Correlation with Yield",
     "Year and pesticide usage carry the strongest (still weak) positive "
     "correlation with yield; temperature trends weakly negative.",
     "No single climate variable can act as a shortcut predictor — planners "
     "need a model that weighs several factors jointly, not a simple rule of thumb."),
    ("crop_yield_by_region.png", "Top 15 Regions by Average Yield",
     "Yield potential is highly geography-dependent — the top regions "
     "consistently out-produce the global average by a wide margin.",
     "Region is a first-order driver of expected output — resource allocation "
     "and yield targets should always be set per-region, not globally."),
    ("top10_regions_yield.png", "Top 10 Regions — Yield Detail",
     "A closer view of the highest-performing regions used to validate "
     "regional stratification in the model.",
     "Confirms the top-performer list is stable, giving stakeholders a "
     "reliable shortlist of benchmark regions for best-practice studies."),
    ("rainfall_vs_yield.png", "Rainfall vs. Crop Yield",
     "The regression trend line is nearly flat — rainfall alone is a weak "
     "linear predictor, motivating the non-linear ensemble approach.",
     "Irrigation planning based on rainfall volume alone would be unreliable — "
     "yield forecasts need the full multi-factor model, not a rainfall lookup."),
    ("temperature_vs_yield.png", "Temperature vs. Crop Yield",
     "A mild negative trend appears, but wide dispersion shows yield is "
     "shaped by many interacting climate and agronomic factors.",
     "Heat-stress risk should be monitored alongside crop and region context, "
     "not treated as a standalone warning signal."),
    ("pairplot.png", "Pairwise Feature Relationships",
     "Weak pairwise linear structure across all numerical drivers supports "
     "using tree-based models capable of capturing non-linear interaction.",
     "Justifies the choice of ensemble tree models over simpler regressions "
     "for any future extension of this analysis."),
]

DIAGNOSTIC_IMAGES = [
    ("model_comparison_cv.png", "5-Fold Cross-Validation Comparison",
     "XGBoost and Random Forest lead the field with average R² above 0.99, "
     "far ahead of the linear baselines.",
     "Ensemble tree models are the right investment for production — linear "
     "baselines would under-serve real forecasting needs."),
    ("hyperparameter_tuning_comparison.png", "Tuning Impact — Before vs. After",
     "RandomizedSearchCV improved both ensemble candidates, reinforcing "
     "tree-based models as the correct family for this problem.",
     "Tuning delivered a measurable accuracy gain at low compute cost — worth "
     "repeating whenever new seasons of data are added."),
    ("residual_plot.png", "Residual Plot",
     "Residuals scatter randomly around zero with no systematic funnel or "
     "curve, indicating an unbiased fit across the prediction range.",
     "The model doesn't systematically over- or under-predict for any yield "
     "band, so its estimates can be trusted across low and high-yield regions alike."),
    ("error_distribution.png", "Prediction Error Distribution",
     "Errors concentrate tightly near zero with only a small tail of larger "
     "residuals, matching the model's strong aggregate R² score.",
     "Most forecasts will land close to actual yield, with only rare outlier "
     "cases needing manual review before decisions are made."),
]

# Shared Plotly color system, matched to the app's palette
BRAND_GREEN_SCALE = ["#0B3D2A", "#14532D", "#1D6E3D", "#2E8B57", "#5FA97C", "#9BC6A8"]
BRAND_GOLD = "#C9A24B"
PLOTLY_FONT = dict(family="Inter, sans-serif", color="#3B4A42", size=13)


# ----------------------------------------------------------------------------
# 2. CACHED DATA / MODEL LOADERS  (UNCHANGED LOGIC)
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
    (UNCHANGED — same logic as the original feature-engineering pass.)
    """
    lookups = {}
    if _df_feat is None or _df_feat.empty:
        return lookups

    if "pesticides_tonnes" in _df_feat.columns:
        try:
            _, bin_edges = pd.qcut(
                _df_feat["pesticides_tonnes"], q=4, retbins=True, duplicates="drop"
            )
            lookups["pesticide_edges"] = bin_edges
        except Exception:
            lookups["pesticide_edges"] = None

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
    (UNCHANGED — identical bins / ratios / proxy logic as before.)
    """
    rain_bins = [0, 500, 1000, 2000, float("inf")]
    rain_labels = ["Low", "Moderate", "High", "Very High"]
    rainfall_category = pd.cut([rainfall], bins=rain_bins, labels=rain_labels)[0]

    gdd = max(avg_temp - 10, 0.0)

    temp_bins = [0, 10, 20, 25, float("inf")]
    temp_labels = ["Cold", "Moderate", "Warm", "Hot"]
    temperature_category = pd.cut([avg_temp], bins=temp_bins, labels=temp_labels)[0]

    edges = lookups.get("pesticide_edges")
    pesticide_labels = ["Low", "Medium", "High", "Very High"]
    if edges is not None and len(edges) >= 2:
        idx = int(np.clip(np.searchsorted(edges, pesticides, side="right") - 1, 0, 3))
        pesticide_category = pesticide_labels[idx]
    else:
        pesticide_category = "Medium"

    rainfall_temp_ratio = rainfall / avg_temp if avg_temp != 0 else 0.0

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
    (estimator disagreement as a practical proxy for uncertainty).
    (UNCHANGED.) Returns (low, high, std) or None.
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


@st.cache_data(show_spinner=False)
def get_live_feature_importance(_pipeline):
    """Read feature_importances_ straight off the already-trained model
    (inference-time introspection only — no fitting happens here)."""
    try:
        preprocessor = _pipeline.named_steps.get("preprocessor")
        model = _pipeline.named_steps.get("model")
        names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_
        fi = pd.DataFrame({"Feature": names, "Importance": importances})
        fi["Feature"] = fi["Feature"].str.replace(r"^(num__|cat__)", "", regex=True)
        return fi.sort_values("Importance", ascending=False).head(10)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 3. LOAD EVERYTHING ONCE  (UNCHANGED)
# ----------------------------------------------------------------------------
df_clean = load_csv(CLEANED_CSV)
df_feat = load_csv(FEATURED_CSV)
model_pipeline = load_model(MODEL_PKL)
eval_results = load_csv(EVAL_CSV)
feature_lookups = build_lookup_tables(df_feat) if df_feat is not None else {}


# ----------------------------------------------------------------------------
# 4. GLOBAL STYLE SYSTEM — premium glass-morphism design language
# ----------------------------------------------------------------------------
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ============================= DESIGN TOKENS ============================= */
:root{
    --forest-950:#071F16;
    --forest-900:#0B3D2A;
    --forest-800:#14532D;
    --forest-700:#1D6E3D;
    --forest-500:#2E8B57;
    --emerald:#10B981;
    --sage-200:#D9E8DC;
    --sage-100:#EEF4EF;
    --gold-500:#C9A24B;
    --gold-600:#B08A34;
    --bg:#F3F6F3;
    --card:#FFFFFF;
    --glass:rgba(255,255,255,0.66);
    --glass-border:rgba(255,255,255,0.55);
    --glass-dark:rgba(9,45,31,0.62);
    --ink-900:#0E1C15;
    --ink-700:#37473F;
    --ink-500:#67766E;
    --line:#E3EAE4;
    --radius-lg:26px;
    --radius-md:20px;
    --radius-sm:13px;
    --shadow-sm:0 1px 2px rgba(14,28,21,0.04), 0 2px 6px rgba(14,28,21,0.05);
    --shadow-md:0 10px 30px rgba(14,28,21,0.10);
    --shadow-lg:0 26px 60px rgba(14,28,21,0.18);
    --shadow-glow:0 0 0 1px rgba(46,139,87,0.18), 0 18px 40px rgba(16,185,129,0.20);
}

html, body, [class*="css"]{ font-family:'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"]{visibility:hidden; height:0;}
.block-container{ padding-top:1.1rem !important; padding-bottom:3rem !important; max-width:1300px; }
body, .stApp{
    background:
        radial-gradient(60rem 30rem at 90% -10%, rgba(16,185,129,0.06) 0%, rgba(16,185,129,0) 60%),
        radial-gradient(50rem 30rem at -10% 10%, rgba(201,162,75,0.07) 0%, rgba(201,162,75,0) 55%),
        var(--bg);
    overflow-x:hidden;
}
/* Layout hardening — grid children default to min-width:auto, which can
   silently force horizontal overflow on laptop-width screens. Pinning it
   to 0 and allowing text to wrap prevents that class of broken layout. */
.kpi-card, .wf-step, .gallery-card, .card, .chart-card, .timeline-card{ min-width:0; word-wrap:break-word; overflow-wrap:break-word; }
img{ max-width:100%; height:auto; }
h1,h2,h3,h4{ font-family:'Sora', sans-serif; color:var(--ink-900); letter-spacing:-0.01em; }
.eyebrow{ font-family:'JetBrains Mono', monospace; font-size:0.72rem; letter-spacing:0.14em;
    text-transform:uppercase; color:var(--forest-700); font-weight:600; }
.muted{ color:var(--ink-500); }

/* ============================== ANIMATIONS ================================ */
@keyframes fadeInUp{
    from{ opacity:0; transform:translateY(16px); }
    to{ opacity:1; transform:translateY(0); }
}
@keyframes floatBlob{
    0%{ transform:translate(0,0) scale(1); }
    50%{ transform:translate(-14px,18px) scale(1.06); }
    100%{ transform:translate(0,0) scale(1); }
}
@keyframes floatBlobSlow{
    0%{ transform:translate(0,0) scale(1); }
    50%{ transform:translate(20px,-16px) scale(1.08); }
    100%{ transform:translate(0,0) scale(1); }
}
@keyframes gradientShift{
    0%{ background-position:0% 50%; }
    50%{ background-position:100% 50%; }
    100%{ background-position:0% 50%; }
}
@keyframes glowPulse{
    0%,100%{ box-shadow:0 0 0 0 rgba(16,185,129,0.28); }
    50%{ box-shadow:0 0 0 8px rgba(16,185,129,0); }
}
@keyframes shimmer{
    0%{ background-position:-400px 0; }
    100%{ background-position:400px 0; }
}
@keyframes ringSpin{
    from{ transform:rotate(0deg); }
    to{ transform:rotate(360deg); }
}
.reveal{ animation:fadeInUp 0.6s cubic-bezier(.22,.9,.32,1) both; }
.reveal-d1{ animation-delay:.05s; } .reveal-d2{ animation-delay:.12s; }
.reveal-d3{ animation-delay:.19s; } .reveal-d4{ animation-delay:.26s; }
.reveal-d5{ animation-delay:.33s; } .reveal-d6{ animation-delay:.40s; }

/* ============================ TOP NAVIGATION =============================== */
.nav-wrap{
    position:sticky; top:10px; z-index:999;
    background:var(--glass-dark);
    backdrop-filter:blur(18px) saturate(140%);
    -webkit-backdrop-filter:blur(18px) saturate(140%);
    border:1px solid rgba(255,255,255,0.10);
    border-radius:999px;
    padding:12px 22px;
    display:flex; align-items:center; justify-content:space-between;
    box-shadow:var(--shadow-lg);
    margin-bottom:30px; flex-wrap:wrap; gap:12px;
}
.brand{ display:flex; align-items:center; gap:10px; color:#FFFFFF;
    font-family:'Sora', sans-serif; font-weight:700; font-size:1.05rem; white-space:nowrap; }
.brand-mark{
    width:32px; height:32px; border-radius:10px;
    background:linear-gradient(135deg, var(--gold-500), var(--emerald));
    background-size:200% 200%; animation:gradientShift 6s ease infinite;
    display:flex; align-items:center; justify-content:center;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,0.3);
}
.nav-links{ display:flex; align-items:center; gap:2px; flex-wrap:wrap; }
.nav-link{
    font-family:'Inter', sans-serif; font-weight:600; font-size:0.85rem;
    color:rgba(255,255,255,0.68); text-decoration:none; padding:9px 18px;
    border-radius:999px; transition:all 0.2s ease; position:relative;
}
.nav-link:hover{ color:#FFFFFF; background:rgba(255,255,255,0.10); }
.nav-link.active{
    color:var(--forest-900); background:#FFFFFF; box-shadow:var(--shadow-sm);
    animation:glowPulse 2.4s ease-in-out infinite;
}

/* ================================ HERO ===================================== */
.hero{
    position:relative; isolation:isolate; overflow:hidden;
    background:linear-gradient(120deg, var(--forest-950) 0%, var(--forest-800) 48%, var(--forest-700) 100%);
    background-size:220% 220%; animation:gradientShift 14s ease infinite;
    border-radius:var(--radius-lg); padding:60px 54px; box-shadow:var(--shadow-lg);
    margin-bottom:30px;
}
.hero-blob{ position:absolute; border-radius:50%; filter:blur(50px); z-index:0; opacity:0.55; }
.hero-blob-1{ width:260px; height:260px; right:-40px; top:-60px;
    background:radial-gradient(circle, rgba(16,185,129,0.55), rgba(16,185,129,0) 70%);
    animation:floatBlob 9s ease-in-out infinite; }
.hero-blob-2{ width:320px; height:320px; right:120px; bottom:-140px;
    background:radial-gradient(circle, rgba(201,162,75,0.45), rgba(201,162,75,0) 70%);
    animation:floatBlobSlow 12s ease-in-out infinite; }
.hero-blob-3{ width:180px; height:180px; left:-40px; bottom:-40px;
    background:radial-gradient(circle, rgba(255,255,255,0.20), rgba(255,255,255,0) 70%);
    animation:floatBlob 10s ease-in-out infinite reverse; }
.hero-ring{ position:absolute; right:60px; top:40px; width:180px; height:180px;
    border:1px dashed rgba(255,255,255,0.18); border-radius:50%; animation:ringSpin 40s linear infinite; z-index:0;}
.hero-content{ position:relative; z-index:2; }
.hero-eyebrow{ font-family:'JetBrains Mono', monospace; color:var(--gold-500); letter-spacing:0.18em;
    text-transform:uppercase; font-size:0.74rem; font-weight:600; }
.hero h1{ color:#FFFFFF; font-size:2.6rem; line-height:1.14; margin:14px 0 14px 0; max-width:700px; }
.hero p{ color:rgba(255,255,255,0.80); font-size:1.03rem; max-width:640px; line-height:1.65; }
.hero-pills{ display:flex; gap:10px; margin-top:24px; flex-wrap:wrap; }
.hero-pill{
    font-size:0.78rem; font-weight:600; color:#FFFFFF;
    background:rgba(255,255,255,0.10); backdrop-filter:blur(6px);
    border:1px solid rgba(255,255,255,0.22); padding:8px 15px; border-radius:999px;
    transition:transform 0.2s ease, background 0.2s ease;
}
.hero-pill:hover{ transform:translateY(-2px); background:rgba(255,255,255,0.18); }

/* ================================ CARDS ===================================== */
.card{
    background:var(--glass); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
    border-radius:var(--radius-md); padding:26px; box-shadow:var(--shadow-sm);
    border:1px solid var(--glass-border); height:100%;
    transition:transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.card:hover{ transform:translateY(-3px); box-shadow:var(--shadow-md); border-color:rgba(46,139,87,0.30); }
.card-title{ font-family:'Sora', sans-serif; font-weight:700; font-size:1.03rem; color:var(--ink-900); margin-bottom:4px;}
.card-sub{ font-size:0.87rem; color:var(--ink-500); margin-bottom:14px; line-height:1.6; }

/* ================================ KPI GRID =================================== */
.kpi-grid{ display:grid; grid-template-columns:repeat(4, 1fr); gap:18px; margin:8px 0 30px 0; }
@media (max-width:1100px){ .kpi-grid{ grid-template-columns:repeat(2, 1fr);} }
.kpi-card{
    position:relative; overflow:hidden;
    background:var(--glass); backdrop-filter:blur(14px);
    border:1px solid var(--glass-border); border-radius:var(--radius-md);
    padding:22px 22px 20px 22px; box-shadow:var(--shadow-sm);
    transition:transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.kpi-card::before{
    content:""; position:absolute; inset:0; border-radius:var(--radius-md);
    padding:1px; background:linear-gradient(135deg, rgba(16,185,129,0), rgba(16,185,129,0));
    transition:background 0.3s ease; -webkit-mask:linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite:xor; mask-composite:exclude; pointer-events:none;
}
.kpi-card:hover{ transform:translateY(-5px) scale(1.01); box-shadow:var(--shadow-glow); border-color:rgba(16,185,129,0.35); }
.kpi-top{ display:flex; align-items:center; justify-content:space-between; margin-bottom:14px;}
.kpi-icon{
    width:40px; height:40px; border-radius:12px;
    background:linear-gradient(135deg, var(--sage-100), var(--sage-200));
    display:flex; align-items:center; justify-content:center;
    transition:transform 0.25s ease;
}
.kpi-card:hover .kpi-icon{ transform:rotate(-8deg) scale(1.08); }
.kpi-tag{ font-family:'JetBrains Mono', monospace; font-size:0.66rem; font-weight:600;
    color:var(--forest-700); background:var(--sage-100); padding:3px 9px; border-radius:999px; }
.kpi-value{ font-family:'Sora', sans-serif; font-size:1.9rem; font-weight:800; color:var(--ink-900); }
.kpi-label{ font-size:0.86rem; color:var(--ink-500); margin-top:4px; font-weight:500;}

/* ============================= WORKFLOW / TIMELINE ============================ */
.workflow{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin:10px 0 8px 0;}
@media (max-width:1100px){ .workflow{ grid-template-columns:repeat(2,1fr);} }
.wf-step{
    background:var(--glass); backdrop-filter:blur(12px); border:1px solid var(--glass-border);
    border-radius:var(--radius-md); padding:20px; box-shadow:var(--shadow-sm); position:relative;
    transition:transform 0.22s ease, box-shadow 0.22s ease;
}
.wf-step:hover{ transform:translateY(-4px); box-shadow:var(--shadow-md); }
.wf-num{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:0.78rem; color:var(--gold-600); margin-bottom:10px; display:block;}
.wf-title{ font-family:'Sora', sans-serif; font-weight:700; font-size:0.95rem; color:var(--ink-900); margin-bottom:6px;}
.wf-desc{ font-size:0.82rem; color:var(--ink-500); line-height:1.5;}

.timeline{ position:relative; padding-left:32px; }
.timeline::before{ content:""; position:absolute; left:9px; top:6px; bottom:6px; width:2px;
    background:linear-gradient(180deg, var(--emerald), var(--gold-500)); border-radius:2px; }
.timeline-item{ position:relative; margin-bottom:22px; }
.timeline-item:last-child{ margin-bottom:0; }
.timeline-dot{ position:absolute; left:-32px; top:4px; width:18px; height:18px; border-radius:50%;
    background:var(--card); border:3px solid var(--forest-600, var(--forest-700)); box-shadow:0 0 0 4px rgba(16,185,129,0.12); }
.timeline-card{ background:var(--glass); backdrop-filter:blur(12px); border:1px solid var(--glass-border);
    border-radius:var(--radius-sm); padding:16px 18px; box-shadow:var(--shadow-sm);
    transition:transform 0.2s ease, box-shadow 0.2s ease; }
.timeline-card:hover{ transform:translateX(4px); box-shadow:var(--shadow-md); }
.timeline-title{ font-family:'Sora', sans-serif; font-weight:700; font-size:0.92rem; color:var(--ink-900); margin-bottom:3px;}
.timeline-desc{ font-size:0.83rem; color:var(--ink-500); line-height:1.55;}

/* =============================== SECTION HEAD ================================= */
.section-head{ display:flex; align-items:baseline; justify-content:space-between; margin:8px 0 18px 0; }
.section-title{ font-family:'Sora', sans-serif; font-weight:700; font-size:1.42rem; color:var(--ink-900);}
.section-desc{ color:var(--ink-500); font-size:0.92rem; margin-top:2px;}

/* ============================== IMAGE GALLERY ================================== */
.gallery-grid{ display:grid; grid-template-columns:repeat(2, 1fr); gap:20px; margin-bottom:8px;}
@media (max-width:900px){ .gallery-grid{ grid-template-columns:1fr;} }
.gallery-card{
    background:var(--card); border:1px solid var(--line); border-radius:var(--radius-md);
    overflow:hidden; box-shadow:var(--shadow-sm);
    transition:transform 0.25s ease, box-shadow 0.25s ease;
}
.gallery-card:hover{ transform:translateY(-5px); box-shadow:var(--shadow-glow); }
.gallery-img-wrap{ overflow:hidden; background:var(--sage-100); }
.gallery-img-wrap img{ width:100%; display:block; transition:transform 0.45s ease; }
.gallery-card:hover .gallery-img-wrap img{ transform:scale(1.04); }
.gallery-body{ padding:18px 20px 20px 20px; }
.gallery-title{ font-family:'Sora', sans-serif; font-weight:700; font-size:1rem; color:var(--ink-900); margin-bottom:6px;}
.gallery-obs{ font-size:0.86rem; color:var(--ink-700); line-height:1.55;}
.gallery-insight{ margin-top:12px; padding:11px 13px; background:var(--sage-100); border-left:3px solid var(--gold-500);
    border-radius:8px; font-size:0.82rem; color:var(--forest-800); line-height:1.5; }
.gallery-insight-tag{ display:block; font-family:'JetBrains Mono', monospace; font-size:0.64rem; font-weight:700;
    letter-spacing:0.08em; text-transform:uppercase; color:var(--gold-600); margin-bottom:4px; }
.gallery-missing{ padding:40px 20px; text-align:center; color:var(--ink-500); font-size:0.86rem; background:var(--sage-100); }

/* ================================== BADGES ===================================== */
.badge-row{ display:flex; flex-wrap:wrap; gap:10px; }
.badge{ background:var(--sage-100); color:var(--forest-800); font-weight:600; font-size:0.82rem;
    padding:8px 16px; border-radius:999px; border:1px solid var(--sage-200);
    transition:transform 0.18s ease, background 0.18s ease; }
.badge:hover{ transform:translateY(-2px); background:var(--sage-200); }

.pill-badge{ display:inline-flex; align-items:center; gap:6px; font-size:0.76rem; font-weight:700;
    padding:6px 13px; border-radius:999px; letter-spacing:0.02em; }
.pill-model{ background:rgba(255,255,255,0.14); color:#fff; border:1px solid rgba(255,255,255,0.25); }
.pill-status-high{ background:rgba(16,185,129,0.18); color:#B6F0D3; border:1px solid rgba(16,185,129,0.35); }
.pill-status-moderate{ background:rgba(201,162,75,0.20); color:#F3DFAE; border:1px solid rgba(201,162,75,0.4); }
.pill-dot{ width:6px; height:6px; border-radius:50%; background:currentColor; display:inline-block; }

/* ============================= RESULT CARD (Prediction) ========================== */
.result-card{
    position:relative; overflow:hidden;
    background:linear-gradient(135deg, var(--forest-950), var(--forest-700));
    background-size:200% 200%; animation:gradientShift 10s ease infinite;
    border-radius:var(--radius-lg); padding:36px 38px; box-shadow:var(--shadow-lg); color:#fff;
}
.result-glow{ position:absolute; width:220px; height:220px; border-radius:50%; right:-60px; top:-60px;
    background:radial-gradient(circle, rgba(16,185,129,0.35), rgba(16,185,129,0) 70%); filter:blur(10px); }
.result-badges{ display:flex; gap:10px; margin-bottom:18px; position:relative; z-index:1; flex-wrap:wrap;}
.result-label{ font-family:'JetBrains Mono', monospace; letter-spacing:0.1em; text-transform:uppercase;
    font-size:0.72rem; color:var(--gold-500); font-weight:700; position:relative; z-index:1;}
.result-value{ font-family:'Sora', sans-serif; font-weight:800; font-size:3.1rem; margin:10px 0 4px 0; position:relative; z-index:1;
    animation:fadeInUp 0.5s ease both; }
.result-unit{ color:rgba(255,255,255,0.65); font-size:0.95rem; position:relative; z-index:1;}
.result-range{ margin-top:20px; padding-top:20px; border-top:1px solid rgba(255,255,255,0.16);
    display:flex; gap:30px; flex-wrap:wrap; position:relative; z-index:1; }
.range-item .k{ font-size:0.75rem; color:rgba(255,255,255,0.6); text-transform:uppercase; letter-spacing:0.08em;}
.range-item .v{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:1.15rem;}

.input-summary-table{ width:100%; border-collapse:collapse; }
.input-summary-table td{ padding:9px 4px; font-size:0.87rem; border-bottom:1px solid var(--line); }
.input-summary-table td.k{ color:var(--ink-500); }
.input-summary-table td.v{ color:var(--ink-900); font-weight:600; text-align:right; font-family:'JetBrains Mono', monospace; }

/* ============================== FORM GROUP LABELS ================================ */
.form-group-label{ font-family:'Sora', sans-serif; font-weight:700; font-size:0.92rem; color:var(--forest-800);
    margin:2px 0 10px 0; display:flex; align-items:center; gap:8px; }
.form-group-label .dot{ width:8px; height:8px; border-radius:50%; background:var(--gold-500); display:inline-block;
    box-shadow:0 0 0 4px rgba(201,162,75,0.18); }

/* ============================ STREAMLIT WIDGET RESTYLE =========================== */
div[data-testid="stSelectbox"] > div, div[data-testid="stNumberInput"] input{ border-radius:var(--radius-sm) !important; }
.stSlider > div > div > div > div{ background:var(--forest-500) !important; }

div.stButton > button{
    position:relative; overflow:hidden; isolation:isolate;
    background:linear-gradient(135deg, var(--forest-700), var(--forest-950));
    background-size:180% 180%;
    color:#fff; border:none; border-radius:999px; font-weight:700;
    padding:0.75rem 1.8rem; box-shadow:var(--shadow-sm); font-family:'Sora', sans-serif;
    transition:transform 0.18s ease, box-shadow 0.18s ease, background-position 0.4s ease;
}
div.stButton > button::after{
    content:""; position:absolute; inset:0; border-radius:999px;
    background:radial-gradient(circle, rgba(255,255,255,0.35) 0%, rgba(255,255,255,0) 60%);
    opacity:0; transform:scale(0.6); transition:opacity 0.5s ease, transform 0.5s ease; z-index:-1;
}
div.stButton > button:active::after{ opacity:1; transform:scale(1.6); transition:opacity 0s, transform 0s; }
div.stButton > button:hover{
    transform:translateY(-2px); box-shadow:var(--shadow-glow); color:#fff; background-position:100% 0%;
}
div.stButton > button p{ color:#fff !important; font-weight:700 !important;}

/* ================================ DATAFRAMES ===================================== */
div[data-testid="stDataFrame"]{ border-radius:var(--radius-md); overflow:hidden; border:1px solid var(--line); }

/* ============================ PLOTLY CONTAINER WRAP =============================== */
.chart-card{ background:var(--glass); backdrop-filter:blur(12px); border:1px solid var(--glass-border);
    border-radius:var(--radius-md); padding:20px 22px 8px 22px; box-shadow:var(--shadow-sm); margin-bottom:22px; }

/* =================================== FOOTER ======================================== */
.app-footer{ margin-top:44px; padding:24px 6px 6px 6px; border-top:1px solid var(--line);
    display:flex; justify-content:space-between; color:var(--ink-500); font-size:0.8rem; flex-wrap:wrap; gap:8px; }

/* ============================== MISSING-DATA NOTICE ================================ */
.notice{ background:#FFF7E8; border:1px solid #F0DDA6; color:#7A5A17;
    border-radius:var(--radius-sm); padding:14px 18px; font-size:0.86rem; margin-bottom:18px; }

/* ============================== SKELETON / SHIMMER ================================= */
.skeleton{ background:linear-gradient(90deg, #EDF2EE 25%, #F7FAF7 37%, #EDF2EE 63%);
    background-size:800px 100%; animation:shimmer 1.6s linear infinite; border-radius:var(--radius-sm); height:180px; }

/* ============================== REQUIREMENTS CHECKLIST ============================== */
.check-grid{ display:grid; grid-template-columns:repeat(2, 1fr); gap:12px; margin-top:6px; }
@media (max-width:900px){ .check-grid{ grid-template-columns:1fr; } }
.check-item{ display:flex; align-items:flex-start; gap:12px; padding:13px 15px; border-radius:var(--radius-sm);
    background:var(--sage-100); border:1px solid var(--sage-200); min-width:0; transition:transform 0.18s ease; }
.check-item:hover{ transform:translateX(3px); }
.check-item.partial{ background:#FFF7E8; border-color:#F0DDA6; }
.check-icon{ width:22px; height:22px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center;
    background:var(--forest-700); color:#fff; font-size:0.72rem; font-weight:800; margin-top:1px; }
.check-item.partial .check-icon{ background:var(--gold-600); }
.check-text .t{ font-weight:700; font-size:0.87rem; color:var(--ink-900); margin-bottom:2px; }
.check-text .d{ font-size:0.78rem; color:var(--ink-500); line-height:1.45; }

/* ================================ IFRAME COMPONENT WRAP ============================== */
.iframe-shell{ margin:8px 0 24px 0; }
</style>
""")


# ----------------------------------------------------------------------------
# 5. SMALL RENDER HELPERS
# ----------------------------------------------------------------------------
def section_header(title: str, desc: str = ""):
    html(
        f"""<div class="section-head reveal">
                <div>
                    <div class="section-title">{title}</div>
                    <div class="section-desc">{desc}</div>
                </div>
            </div>"""
    )


def kpi_card_html(icon_svg: str, tag: str, value: str, label: str, delay_cls: str = "") -> str:
    return f"""
    <div class="kpi-card reveal {delay_cls}">
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


def base_plotly_layout(fig, height=360):
    """Apply a single, consistent premium theme to every Plotly figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        title_font=dict(family="Sora, sans-serif", size=15, color="#0E1C15"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11)),
        hoverlabel=dict(bgcolor="#0B3D2A", font_color="white", font_family="Inter"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E9EFE9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E9EFE9", zeroline=False)
    return fig


def render_gallery(image_specs, base_dir: Path, columns: int = 2):
    """Render a responsive card gallery for a list of (file, title, observation, insight)."""
    cards_html = []
    for i, (filename, title, observation, insight) in enumerate(image_specs):
        path = base_dir / filename
        b64 = image_to_base64(path)
        if b64:
            img_block = f'<div class="gallery-img-wrap"><img src="{b64}" /></div>'
        else:
            img_block = f'<div class="gallery-missing">Image not found: {filename}</div>'
        delay = f"reveal-d{(i % 6) + 1}"
        cards_html.append(f"""
        <div class="gallery-card reveal {delay}">
            {img_block}
            <div class="gallery-body">
                <div class="gallery-title">{title}</div>
                <div class="gallery-obs">{observation}</div>
                <div class="gallery-insight"><span class="gallery-insight-tag">Business Insight</span>{insight}</div>
            </div>
        </div>
        """)
    grid_style = f'grid-template-columns:repeat({columns}, 1fr);'
    html(
        f'<div class="gallery-grid" style="{grid_style}">{"".join(cards_html)}</div>'
    )


def missing_file_notice(paths_and_labels):
    missing = [label for path, label in paths_and_labels if not Path(path).exists()]
    if missing:
        html(
            f'<div class="notice">⚠ Some expected files were not found on disk and '
            f'related sections are showing limited data: {", ".join(missing)}.</div>'
        )


def render_animated_kpis(kpis, height: int = 190):
    """
    Render the hero KPI row as genuinely animated count-up numbers.

    This intentionally uses st.iframe rather than st.markdown:
    a Streamlit component runs inside its own sandboxed iframe, so a
    <script> placed there actually executes (unlike a <script> tag passed
    to st.markdown, which the browser will not run). All markup, styling
    and script for this widget are self-contained in the same iframe —
    that avoids the cross-frame DOM-access restriction that would occur
    if the script tried to reach elements rendered in the main page.

    Each kpi tuple is (tag, target_value, label, is_numeric, decimals).
    """
    cards = []
    for i, (tag, target, label, is_numeric, decimals) in enumerate(kpis):
        display_target = target if is_numeric else 0
        static_text = "" if is_numeric else target
        cards.append(f"""
        <div class="k-card" style="animation-delay:{i * 90}ms">
            <div class="k-tag">{tag}</div>
            <div class="k-value" data-target="{display_target}" data-decimals="{decimals}"
                 data-static="{static_text}">0</div>
            <div class="k-label">{label}</div>
        </div>
        """)

    component_html = f"""
    <div class="k-wrap">
        <style>
            * {{ box-sizing:border-box; }}
            body {{ margin:0; background:transparent;
                font-family:'Inter','Sora',-apple-system,sans-serif; }}
            .k-wrap {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:18px; }}
            @media (max-width:1000px) {{ .k-wrap {{ grid-template-columns:repeat(2, 1fr); }} }}
            .k-card {{ background:rgba(255,255,255,0.7); border:1px solid rgba(255,255,255,0.6);
                border-radius:20px; padding:20px; box-shadow:0 1px 2px rgba(14,28,21,0.04), 0 2px 6px rgba(14,28,21,0.05);
                opacity:0; transform:translateY(14px); animation:kFadeUp 0.6s cubic-bezier(.22,.9,.32,1) forwards; }}
            @keyframes kFadeUp {{ to {{ opacity:1; transform:translateY(0); }} }}
            .k-tag {{ font-family:'JetBrains Mono',monospace; font-size:0.66rem; font-weight:700; letter-spacing:0.08em;
                color:#14532D; background:#EEF4EF; display:inline-block; padding:3px 9px; border-radius:999px; margin-bottom:12px; }}
            .k-value {{ font-family:'Sora',sans-serif; font-size:1.9rem; font-weight:800; color:#0E1C15; }}
            .k-label {{ font-size:0.86rem; color:#67766E; margin-top:4px; font-weight:500; }}
        </style>
        {"".join(cards)}
        <script>
            const els = document.querySelectorAll('.k-value');
            els.forEach((el) => {{
                const staticText = el.getAttribute('data-static');
                if (staticText) {{ el.textContent = staticText; return; }}
                const target = parseFloat(el.getAttribute('data-target'));
                const decimals = parseInt(el.getAttribute('data-decimals'));
                const duration = 1100;
                const start = performance.now();
                function tick(now) {{
                    const progress = Math.min((now - start) / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    const current = target * eased;
                    el.textContent = current.toLocaleString(undefined, {{
                        minimumFractionDigits: decimals, maximumFractionDigits: decimals
                    }});
                    if (progress < 1) {{ requestAnimationFrame(tick); }}
                }}
                requestAnimationFrame(tick);
            }});
        </script>
    </div>
    """
    st.iframe(component_html, height=height)


def render_requirements_checklist(items):
    """
    Render the Code-A-Nova requirement checklist. `items` is a list of
    (title, description, status) where status is 'done' or 'partial'.
    Statuses are reported honestly — substituted features (e.g. no raw
    NPK/humidity columns existed in the source dataset) are marked
    'partial' with an explanation rather than claimed as fully met.
    """
    cards = []
    for title, desc, status in items:
        cls = "partial" if status == "partial" else ""
        icon = "~" if status == "partial" else "✓"
        cards.append(f"""
        <div class="check-item {cls}">
            <div class="check-icon">{icon}</div>
            <div class="check-text">
                <div class="t">{title}</div>
                <div class="d">{desc}</div>
            </div>
        </div>
        """)
    html('<div class="check-grid">' + "".join(cards) + "</div>")


# ----------------------------------------------------------------------------
# 6. NAVIGATION — floating glass top bar, no sidebar, no Streamlit tabs
# ----------------------------------------------------------------------------
NAV_ITEMS = [
    ("dashboard", "Dashboard"),
    ("eda", "Exploratory Data Analysis"),
    ("prediction", "Crop Yield Prediction"),
    ("performance", "Model Performance"),
    ("about", "About Project"),
]

query_params = st.query_params
active_page = query_params.get("page", "dashboard")
if active_page not in dict(NAV_ITEMS):
    active_page = "dashboard"

nav_links_html = "".join(
    f'<a class="nav-link {"active" if key == active_page else ""}" href="?page={key}">{label}</a>'
    for key, label in NAV_ITEMS
)

html(
    f"""
    <div class="nav-wrap">
        <div class="brand">
            <div class="brand-mark">{ICON_LEAF.replace('#14532D','#FFFFFF')}</div>
            CropYield Predictor
        </div>
        <div class="nav-links">{nav_links_html}</div>
    </div>
    """
)


# ----------------------------------------------------------------------------
# 7. PAGE — DASHBOARD (HOME)
# ----------------------------------------------------------------------------
def page_dashboard():
    html(
        """
        <div class="hero reveal">
            <div class="hero-blob hero-blob-1"></div>
            <div class="hero-blob hero-blob-2"></div>
            <div class="hero-blob hero-blob-3"></div>
            <div class="hero-ring"></div>
            <div class="hero-content">
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
        </div>
        """
    )

    missing_file_notice([
        (CLEANED_CSV, "cleaned_crop_yield.csv"),
        (MODEL_PKL, "random_forest_pipeline.pkl"),
        (EVAL_CSV, "model_evaluation_results.csv"),
    ])

    if df_clean is not None:
        n_rows_val, n_rows_numeric = len(df_clean), True
        if "Area" in df_clean.columns:
            n_regions_val, n_regions_numeric = int(df_clean["Area"].nunique()), True
        else:
            n_regions_val, n_regions_numeric = "—", False
        if "Item" in df_clean.columns:
            n_crops_val, n_crops_numeric = int(df_clean["Item"].nunique()), True
        else:
            n_crops_val, n_crops_numeric = "—", False
        year_range = (
            f"{int(df_clean['Year'].min())}–{int(df_clean['Year'].max())}"
            if "Year" in df_clean.columns else "—"
        )
    else:
        n_rows_val, n_rows_numeric = "—", False
        n_regions_val, n_regions_numeric = "—", False
        n_crops_val, n_crops_numeric = "—", False
        year_range = "—"

    if eval_results is not None and "R2 Score" in eval_results.columns:
        best_r2_val, best_r2_numeric = float(eval_results["R2 Score"].max()), True
    else:
        best_r2_val, best_r2_numeric = "—", False

    # (tag, target_value, label, is_numeric, decimals)
    animated_kpis = [
        ("RECORDS", n_rows_val, "Historical observations", n_rows_numeric, 0),
        ("COVERAGE", n_regions_val, "Countries / regions tracked", n_regions_numeric, 0),
        ("CROPS", n_crops_val, "Distinct crop types", n_crops_numeric, 0),
        ("BEST MODEL", best_r2_val, "Highest test R² score", best_r2_numeric, 3),
    ]
    render_animated_kpis(animated_kpis)

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        html('<div class="card reveal reveal-d2">')
        html('<div class="card-title">Project Summary</div>')
        html(
            """<div class="card-sub">
            This platform analyzes historical weather, rainfall, pesticide usage and
            temperature data across multiple regions and crop types to predict agricultural
            yield (measured in hectograms per hectare). The workflow spans data validation,
            cleaning, exploratory analysis, domain-driven feature engineering, multi-model
            comparison under 5-fold cross-validation, hyperparameter tuning via
            RandomizedSearchCV, and final deployment of a tuned Random Forest pipeline.
            </div>"""
        )
        html("</div>")
        st.write("")

        html('<div class="section-title" style="font-size:1.1rem;">Workflow</div>')
        steps = [
            ("01", "Data Validation", "Schema checks, missing-value and duplicate audits on raw records."),
            ("02", "Cleaning", "Outlier review via IQR and dtype correction, exported as a clean dataset."),
            ("03", "EDA", "Univariate, bivariate and correlation analysis across all key drivers."),
            ("04", "Feature Engineering", "Rainfall/temperature bins, GDD, pesticide quartiles and ratio features."),
            ("05", "Modeling & Tuning", "5 models compared via CV; Random Forest and XGBoost tuned further."),
        ]
        html(
            '<div class="workflow">' + "".join(
                f"""<div class="wf-step reveal reveal-d{i+1}">
                        <span class="wf-num">{num}</span>
                        <div class="wf-title">{title}</div>
                        <div class="wf-desc">{desc}</div>
                    </div>"""
                for i, (num, title, desc) in enumerate(steps)
            ) + "</div>"
        )

    with right:
        html('<div class="card reveal reveal-d3">')
        html('<div class="card-title">Dataset Overview</div>')
        html(f'<div class="card-sub">Coverage window: {year_range}</div>')
        if df_clean is not None:
            preview_cols = [c for c in ["Area", "Item", "Year", "hg/ha_yield",
                                         "average_rain_fall_mm_per_year",
                                         "pesticides_tonnes", "avg_temp"] if c in df_clean.columns]
            st.dataframe(df_clean[preview_cols].head(8), width="stretch", hide_index=True)
        else:
            st.info("Cleaned dataset not found on disk yet.")
        html("</div>")

        st.write("")
        html('<div class="card reveal reveal-d4">')
        html('<div class="card-title">Quick Insights</div>')
        insight_list = [
            "Ensemble tree models (Random Forest, XGBoost) outperform linear baselines by a wide margin.",
            "Yield varies far more by region and crop type than by any single climate variable.",
            "Rainfall and temperature alone are weak linear predictors — non-linear interaction matters.",
        ]
        html(
            "".join(f'<p class="card-sub" style="margin-bottom:10px;">▸ {t}</p>' for t in insight_list)
        )
        html("</div>")

    # ---- Interactive Plotly charts, computed live from the cleaned dataset ----
    if df_clean is not None and {"Year", TARGET_COL}.issubset(df_clean.columns):
        st.write("")
        html('<div class="section-title" style="font-size:1.15rem; margin-bottom:14px;">Interactive Charts</div>')
        chart_left, chart_right = st.columns(2, gap="large")

        with chart_left:
            html('<div class="chart-card reveal">')
            yearly = df_clean.groupby("Year")[TARGET_COL].mean().reset_index()
            fig1 = px.area(
                yearly, x="Year", y=TARGET_COL,
                title="Average Crop Yield Over Time",
                color_discrete_sequence=[BRAND_GREEN_SCALE[2]],
            )
            fig1.update_traces(line=dict(width=3), fillcolor="rgba(46,139,87,0.14)")
            base_plotly_layout(fig1)
            st.plotly_chart(fig1, width="stretch", config={"displayModeBar": False})
            html("</div>")

        with chart_right:
            html('<div class="chart-card reveal reveal-d2">')
            if "Area" in df_clean.columns:
                top_regions = (
                    df_clean.groupby("Area")[TARGET_COL].mean()
                    .sort_values(ascending=False).head(10).reset_index()
                )
                fig2 = px.bar(
                    top_regions, x=TARGET_COL, y="Area", orientation="h",
                    title="Top 10 Regions by Average Yield",
                    color=TARGET_COL, color_continuous_scale=BRAND_GREEN_SCALE,
                )
                fig2.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
                base_plotly_layout(fig2)
                st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
            html("</div>")


# ----------------------------------------------------------------------------
# 8. PAGE — EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------------
def page_eda():
    section_header(
        "Exploratory Data Analysis",
        "Distribution, correlation and regional insight visualizations generated directly from the notebook."
    )
    missing_file_notice([(IMAGES_DIR / f, f) for f, _, _, _ in EDA_IMAGES])
    render_gallery(EDA_IMAGES, IMAGES_DIR, columns=2)


# ----------------------------------------------------------------------------
# 9. PAGE — CROP YIELD PREDICTION
# ----------------------------------------------------------------------------
def page_prediction():
    section_header(
        "Crop Yield Prediction",
        "Generate a live estimate from the tuned Random Forest pipeline — inference only, no retraining."
    )

    if model_pipeline is None:
        html(
            '<div class="notice">⚠ The trained pipeline '
            f'(<code>{MODEL_PKL.name}</code>) was not found. Place it under '
            '<code>models/</code> to enable predictions.</div>'
        )
        return

    areas = sorted(df_feat["Area"].dropna().unique().tolist()) if df_feat is not None and "Area" in df_feat.columns else ["Unknown"]
    items = sorted(df_feat["Item"].dropna().unique().tolist()) if df_feat is not None and "Item" in df_feat.columns else ["Unknown"]

    form_left, form_right = st.columns(2, gap="large")

    with form_left:
        html('<div class="card reveal">')
        html('<div class="form-group-label"><span class="dot"></span>Location &amp; Crop</div>')
        area = st.selectbox("Region", areas, index=0)
        item = st.selectbox("Crop Type", items, index=0)
        year = st.slider("Year", min_value=1990, max_value=2035, value=2024, step=1)
        html("</div>")

    with form_right:
        html('<div class="card reveal reveal-d2">')
        html('<div class="form-group-label"><span class="dot"></span>Climate &amp; Inputs</div>')
        rainfall = st.number_input("Average Rainfall (mm / year)", min_value=0.0, max_value=5000.0, value=1100.0, step=10.0)
        avg_temp = st.number_input("Average Temperature (°C)", min_value=-10.0, max_value=45.0, value=20.5, step=0.1)
        pesticides = st.number_input("Pesticide Usage (tonnes)", min_value=0.0, max_value=400000.0, value=17000.0, step=100.0)
        html("</div>")

    st.write("")
    predict_clicked = st.button("Predict Crop Yield")

    if predict_clicked:
        input_row = engineer_features(area, item, year, rainfall, pesticides, avg_temp, feature_lookups)
        try:
            prediction = float(model_pipeline.predict(input_row)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        spread = tree_ensemble_spread(model_pipeline, input_row)

        # Confidence status badge derived from relative tree-ensemble spread.
        status_label, status_cls = "High Confidence", "pill-status-high"
        if spread:
            _, _, std = spread
            relative_spread = std / prediction if prediction else 1.0
            if relative_spread > 0.12:
                status_label, status_cls = "Moderate Confidence", "pill-status-moderate"

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
            html(
                f"""
                <div class="result-card reveal">
                    <div class="result-glow"></div>
                    <div class="result-badges">
                        <span class="pill-badge pill-model">Random Forest · Tuned</span>
                        <span class="pill-badge {status_cls}"><span class="pill-dot"></span>{status_label}</span>
                    </div>
                    <div class="result-label">Predicted Crop Yield</div>
                    <div class="result-value">{prediction:,.0f}</div>
                    <div class="result-unit">hectograms per hectare (hg/ha)</div>
                    {range_html}
                </div>
                """
            )
            st.caption(
                "Confidence range is derived from disagreement across the Random Forest's "
                "individual decision trees (10th–90th percentile of tree-level predictions), "
                "not a formal statistical confidence interval."
            )

        with summary_col:
            html('<div class="card reveal reveal-d2">')
            html('<div class="card-title">Input Summary</div>')
            summary_rows = [
                ("Region", area), ("Crop", item), ("Year", year),
                ("Rainfall", f"{rainfall:,.0f} mm/yr"),
                ("Avg. Temperature", f"{avg_temp:.1f} °C"),
                ("Pesticide Usage", f"{pesticides:,.0f} t"),
            ]
            html(
                '<table class="input-summary-table">' + "".join(
                    f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>' for k, v in summary_rows
                ) + "</table>"
            )
            html("</div>")


# ----------------------------------------------------------------------------
# 10. PAGE — MODEL PERFORMANCE
# ----------------------------------------------------------------------------
def page_performance():
    section_header(
        "Model Performance",
        "Cross-validated comparison, tuning impact and diagnostic plots for the final Random Forest pipeline."
    )
    missing_file_notice([(EVAL_CSV, "model_evaluation_results.csv")])

    if eval_results is not None and "R2 Score" in eval_results.columns:
        best_row = eval_results.sort_values("R2 Score", ascending=False).iloc[0]
        kpis = [
            (ICON_TARGET, "BEST MODEL", str(best_row["Model"]), "Top performer on held-out test set"),
            (ICON_TREND, "R² SCORE", f"{best_row['R2 Score']:.4f}", "Variance explained"),
            (ICON_GRID, "MAE", f"{best_row['MAE']:,.0f}", "Mean absolute error (hg/ha)"),
            (ICON_CLOCK, "RMSE", f"{best_row['RMSE']:,.0f}", "Root mean squared error (hg/ha)"),
        ]
        html(
            '<div class="kpi-grid">' + "".join(
                kpi_card_html(*k, delay_cls=f"reveal-d{i+1}") for i, k in enumerate(kpis)
            ) + "</div>"
        )

        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            html('<div class="chart-card reveal">')
            comp = eval_results.sort_values("R2 Score", ascending=True)
            fig_cmp = go.Figure(go.Bar(
                x=comp["R2 Score"], y=comp["Model"], orientation="h",
                marker=dict(color=comp["R2 Score"], colorscale=BRAND_GREEN_SCALE),
                text=[f"{v:.3f}" for v in comp["R2 Score"]], textposition="outside",
            ))
            fig_cmp.update_layout(title="Model Comparison — R² Score")
            base_plotly_layout(fig_cmp)
            st.plotly_chart(fig_cmp, width="stretch", config={"displayModeBar": False})
            html("</div>")

        with chart_right:
            html('<div class="chart-card reveal reveal-d2">')
            fi = get_live_feature_importance(model_pipeline) if model_pipeline is not None else None
            if fi is not None and not fi.empty:
                fig_fi = px.bar(
                    fi.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                    title="Top 10 Feature Importances (Random Forest)",
                    color="Importance", color_continuous_scale=[BRAND_GOLD, BRAND_GREEN_SCALE[2]],
                )
                fig_fi.update_layout(coloraxis_showscale=False)
                base_plotly_layout(fig_fi)
                st.plotly_chart(fig_fi, width="stretch", config={"displayModeBar": False})
            else:
                st.info("Feature importance unavailable — model not loaded.")
            html("</div>")

        html('<div class="card reveal">')
        html('<div class="card-title">Model Comparison Table</div>')
        html('<div class="card-sub">Evaluated on the held-out test split.</div>')
        display_df = eval_results.sort_values("R2 Score", ascending=False).reset_index(drop=True)
        st.dataframe(
            display_df.style.format({"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "R2 Score": "{:.4f}"}),
            width="stretch", hide_index=True,
        )
        html("</div>")
        st.write("")
    else:
        st.info("Model evaluation results not found on disk yet.")

    html('<div class="section-title" style="font-size:1.15rem; margin:22px 0 14px 0;">Diagnostic Snapshots</div>')
    missing_file_notice([(IMAGES_DIR / f, f) for f, _, _, _ in DIAGNOSTIC_IMAGES])
    render_gallery(DIAGNOSTIC_IMAGES, IMAGES_DIR, columns=2)


# ----------------------------------------------------------------------------
# 11. PAGE — ABOUT PROJECT
# ----------------------------------------------------------------------------
def page_about():
    section_header("About This Project", "Scope, pipeline architecture and technology stack.")

    html('<div class="card reveal">')
    html('<div class="card-title">Overview</div>')
    html(
        """<div class="card-sub">
        CropYield Predictor is an end-to-end agricultural analytics platform built to
        forecast crop yield from historical rainfall, temperature and pesticide-usage
        records. The project follows a fully reproducible Scikit-Learn pipeline —
        preprocessing and modeling live in a single serialized object, eliminating data
        leakage between training and inference.
        </div>"""
    )
    html("</div>")
    st.write("")

    html('<div class="card reveal reveal-d2">')
    html('<div class="card-title">Project Requirements Checklist</div>')
    html('<div class="card-sub">Mapped against the Code-A-Nova Task 4/5 brief. Items marked '
         '"Adapted" reflect a substituted feature where the source dataset did not contain '
         'the originally-specified raw column.</div>')
    requirement_items = [
        ("Data Ingestion &amp; Validation", "Schema checks, missing-value and duplicate audits on raw records.", "done"),
        ("Data Cleaning", "IQR-based outlier review and dtype correction, exported as a clean dataset.", "done"),
        ("EDA — 15+ Visualizations", "Univariate, bivariate, correlation and pairwise plots across all key drivers.", "done"),
        ("Rainfall Category Binning", "Low / Moderate / High / Very High bands engineered from rainfall volume.", "done"),
        ("Growing Degree Days", "Computed from average temperature with a 10°C base, as specified.", "done"),
        ("NPK Ratios &amp; Humidity Index", "Source dataset has no N/P/K or humidity columns — substituted with "
         "pesticide-usage quartiles and a rainfall/temperature ratio as equivalent engineered signals.", "partial"),
        ("Scikit-Learn Pipeline", "ColumnTransformer combining StandardScaler (numeric) and OneHotEncoder "
         "(categorical) in one reproducible object — no data leakage.", "done"),
        ("5+ Model Architectures", "Linear Regression, Ridge, Random Forest, Gradient Boosting and XGBoost compared.", "done"),
        ("5-Fold Cross-Validation", "All models evaluated under 5-fold CV rather than a single train/test split.", "done"),
        ("RandomizedSearchCV Tuning", "Hyperparameter search applied to the top two ensemble candidates.", "done"),
        ("Feature Importance &amp; Residuals", "Random Forest importances, residual scatter and error-distribution "
         "analysis all included.", "done"),
        ("Interactive Prediction Dashboard", "Live inference against the tuned pipeline with confidence range and "
         "input summary.", "done"),
    ]
    render_requirements_checklist(requirement_items)
    html("</div>")
    st.write("")

    html('<div class="card reveal reveal-d3">')
    html('<div class="card-title">Learning Outcomes</div>')
    outcomes = [
        "Executed a complete, reproducible data science project from raw ingestion to a deployed dashboard.",
        "Applied systematic EDA to uncover patterns, outliers and feature relationships.",
        "Engineered domain-relevant features that improve model performance through applied agronomic knowledge.",
        "Built a Scikit-Learn Pipeline combining preprocessing and modeling for reproducible ML workflows.",
        "Compared multiple regression models using cross-validation and selected the best on multiple metrics.",
        "Performed hyperparameter tuning with RandomizedSearchCV and interpreted its effect on generalization.",
        "Communicated findings through professional visualization and a structured, interactive report.",
        "Built an interactive dashboard allowing non-technical stakeholders to explore predictions directly.",
    ]
    html("".join(f'<p class="card-sub" style="margin-bottom:9px;">▸ {o}</p>' for o in outcomes))
    html("</div>")
    st.write("")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        html('<div class="card reveal reveal-d2">')
        html('<div class="card-title">Pipeline Architecture — Timeline</div>')
        stages = [
            ("Ingestion &amp; Validation", "Raw CSV ingestion with schema and missing-value validation."),
            ("Cleaning", "Duplicate removal, IQR outlier review, dtype correction."),
            ("Feature Engineering", "Rainfall/temperature bins, Growing Degree Days, pesticide "
             "quartiles, rainfall/temperature ratio, yield-efficiency ratio."),
            ("Preprocessing", "ColumnTransformer — StandardScaler (numeric) + OneHotEncoder (categorical)."),
            ("Model Comparison", "Linear Regression, Ridge, Random Forest, Gradient Boosting, "
             "XGBoost — all under 5-fold cross-validation."),
            ("Hyperparameter Tuning", "RandomizedSearchCV on the top two ensemble candidates."),
            ("Export", "Final tuned Random Forest pipeline serialized via joblib."),
        ]
        timeline_html = '<div class="timeline">' + "".join(
            f"""<div class="timeline-item">
                    <div class="timeline-dot"></div>
                    <div class="timeline-card">
                        <div class="timeline-title">{title}</div>
                        <div class="timeline-desc">{desc}</div>
                    </div>
                </div>"""
            for title, desc in stages
        ) + "</div>"
        html(timeline_html)
        html("</div>")

    with col_b:
        html('<div class="card reveal reveal-d3">')
        html('<div class="card-title">Technology Stack</div>')
        badges = ["Python", "Pandas", "NumPy", "Scikit-Learn", "XGBoost",
                   "Matplotlib", "Seaborn", "Plotly", "Streamlit", "Joblib"]
        html(
            '<div class="badge-row">' + "".join(f'<div class="badge">{b}</div>' for b in badges) + "</div>"
        )
        st.write("")
        html('<div class="card-title" style="margin-top:8px;">Dataset</div>')
        html(
            """<div class="card-sub">
            Historical crop yield records spanning multiple countries and crop types,
            with rainfall, pesticide usage and average temperature as core predictors.
            </div>"""
        )
        html("</div>")

        st.write("")
        html('<div class="card reveal reveal-d4">')
        html('<div class="card-title">Future Scope</div>')
        future = [
            "Incorporate soil nutrient (N-P-K) and humidity sensor data as they become available.",
            "Add MLflow experiment tracking for full run-history comparison inside the dashboard.",
            "Extend to multi-season, multi-year forecasting with time-aware validation.",
            "Publish a public API endpoint for programmatic yield predictions.",
        ]
        html(
            "".join(f'<p class="card-sub" style="margin-bottom:10px;">▸ {t}</p>' for t in future)
        )
        html("</div>")


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
html(
    """
    <div class="app-footer">
        <div>CropYield Predictor · Agricultural Intelligence &amp; Analytics Platform</div>
        <div>Built on a reproducible Scikit-Learn pipeline · No retraining performed at runtime</div>
    </div>
    """
)