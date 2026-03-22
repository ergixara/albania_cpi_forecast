"""
================================================================================
ALBANIA CPI FORECASTING — STREAMLIT WEB APPLICATION
================================================================================
Interactive dashboard wrapping the full ARIMA / Prophet / XGBoost pipeline.
Run with:   streamlit run app.py
================================================================================
"""

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Tuple, Dict, Optional
import io

# ── ML libs ──────────────────────────────────────────────────────────────────
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from xgboost import XGBRegressor

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Albania CPI Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #0f172a;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; }

/* ── hero header ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(56,189,248,.12) 0%, transparent 70%);
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem; font-weight: 400;
    color: #f8fafc; margin: 0 0 .5rem;
    line-height: 1.1;
}
.hero h1 em { font-style: italic; color: #38bdf8; }
.hero p { color: #94a3b8; font-size: 1rem; margin: 0; }
.hero .badge {
    display: inline-block;
    background: rgba(56,189,248,.15); border: 1px solid rgba(56,189,248,.3);
    color: #38bdf8; font-family: 'JetBrains Mono', monospace;
    font-size: .72rem; padding: .25rem .7rem; border-radius: 999px;
    margin-right: .5rem; margin-top: .8rem;
}

/* ── metric cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0; }
.metric-card {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.25rem 1.5rem;
    transition: box-shadow .2s;
}
.metric-card:hover { box-shadow: 0 4px 20px rgba(15,23,42,.08); }
.metric-card .label {
    font-size: .72rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #64748b; margin-bottom: .3rem;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem; font-weight: 600; color: #0f172a;
}
.metric-card .delta { font-size: .8rem; color: #64748b; margin-top: .2rem; }
.metric-card.highlight { background: #0f172a; border-color: #0f172a; }
.metric-card.highlight .label { color: #38bdf8; }
.metric-card.highlight .value { color: #f8fafc; }
.metric-card.highlight .delta { color: #94a3b8; }

/* ── section heading ── */
.section-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem; color: #0f172a;
    margin: 2rem 0 1rem;
    padding-bottom: .5rem;
    border-bottom: 2px solid #e2e8f0;
}
.section-heading span { color: #38bdf8; }

/* ── winner badge ── */
.winner-badge {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    color: #38bdf8; font-family: 'JetBrains Mono', monospace;
    font-size: .75rem; font-weight: 600;
    padding: .3rem .9rem; border-radius: 999px;
    display: inline-block; margin-left: .7rem;
    vertical-align: middle;
}

/* ── table override ── */
.dataframe thead tr th {
    background: #0f172a !important; color: #f8fafc !important;
    font-family: 'JetBrains Mono', monospace; font-size: .78rem;
}
.dataframe tbody tr:nth-child(even) { background: #f8fafc; }

/* ── info box ── */
.info-box {
    background: #f0f9ff; border-left: 3px solid #38bdf8;
    padding: .9rem 1.2rem; border-radius: 0 8px 8px 0;
    font-size: .88rem; color: #0f172a; margin: 1rem 0;
}
.info-box strong { color: #0369a1; }

/* ── step pill ── */
.step-pill {
    display: inline-flex; align-items: center; gap: .4rem;
    background: #0f172a; color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem; padding: .2rem .7rem;
    border-radius: 999px; margin-bottom: .6rem;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }

/* ── upload zone ── */
.upload-zone {
    border: 2px dashed #cbd5e1; border-radius: 12px;
    padding: 2.5rem; text-align: center;
    background: #f8fafc; margin: 1rem 0;
}
.upload-zone h3 { font-family: 'DM Serif Display', serif; color: #0f172a; }
.upload-zone p  { color: #64748b; font-size: .9rem; }

/* ── progress step ── */
.progress-step {
    display: flex; align-items: center; gap: .8rem;
    padding: .6rem 1rem; border-radius: 8px;
    background: #f8fafc; margin-bottom: .5rem;
    font-size: .9rem;
}
.progress-step .icon { font-size: 1.1rem; }
.progress-step.done { background: #f0fdf4; color: #166534; }
.progress-step.active { background: #eff6ff; color: #1e40af; }
.progress-step.waiting { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE (Plotly)
# ══════════════════════════════════════════════════════════════════════════════
CLR = {
    "sarima":         "#e11d48",
    "boosted_sarima": "#f97316",
    "prophet":        "#8b5cf6",
    "prophet_boost":  "#38bdf8",
    "actual":         "#0f172a",
    "historical":     "#64748b",
    "fill":           "rgba(56,189,248,.15)",
}
PLOTLY_LAYOUT = dict(
    font_family="DM Sans",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(255,255,255,.8)", bordercolor="#e2e8f0", borderwidth=1),
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean_wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    dates = df_wide.iloc[3, 1:].dropna().values
    date_list = []
    for d in dates:
        try:
            date_list.append(pd.to_datetime(d, format="%Y-%m"))
        except Exception:
            date_list.append(pd.NaT)

    data_rows = []
    for idx in range(4, len(df_wide)):
        cell = df_wide.iloc[idx, 0]
        if pd.isna(cell) or str(cell) == "nan":
            continue
        category_full = str(cell)
        if " " in category_full:
            code = category_full.split(" ")[0]
            name = category_full[len(code):].strip()
        else:
            code = name = category_full
        vals = df_wide.iloc[idx, 1: len(dates) + 1].values
        for dt, v in zip(date_list, vals):
            if pd.notna(dt) and pd.notna(v):
                try:
                    data_rows.append({"Date": dt, "Category": name, "Category_Code": code, "CPI": float(v)})
                except (ValueError, TypeError):
                    pass

    df = pd.DataFrame(data_rows)
    return df.sort_values(["Category_Code", "Date"]).reset_index(drop=True)


def get_series(df_long: pd.DataFrame, code: str = "000000") -> pd.DataFrame:
    s = df_long[df_long["Category_Code"] == code][["Date", "CPI"]]
    return s.sort_values("Date").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, ratio: float = 0.8):
    n = len(df)
    k = int(n * ratio)
    return df.iloc[:k].copy(), df.iloc[k:].copy()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CLASSES  (identical logic to the original pipeline)
# ══════════════════════════════════════════════════════════════════════════════

class SARIMAForecaster:
    def __init__(self, order=(0,1,0), seasonal_order=(1,1,2,12)):
        self.order = order; self.seasonal_order = seasonal_order
        self.fitted_model = self.fitted_values = self.residuals = None

    def fit(self, df):
        y = df["CPI"].values
        m = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
        self.fitted_model = m.fit(disp=False)
        self.fitted_values = self.fitted_model.fittedvalues
        self.residuals = self.fitted_model.resid
        return self

    def predict(self, steps):
        return self.fitted_model.get_forecast(steps=steps).predicted_mean

    def aic(self): return self.fitted_model.aic
    def bic(self): return self.fitted_model.bic


class BoostedSARIMA:
    def __init__(self, sarima_order=(1,1,1), sarima_seasonal=(0,1,1,12), xgb_params=None):
        self.sarima = SARIMAForecaster(order=sarima_order, seasonal_order=sarima_seasonal)
        self.xgb = XGBRegressor(**(xgb_params or {
            "n_estimators":100,"max_depth":5,"learning_rate":.1,
            "subsample":.8,"colsample_bytree":.8,"random_state":42,"objective":"reg:squarederror"}))
        self.feature_cols = None

    def _feats(self, df):
        d = df.copy()
        d["year"] = d["Date"].dt.year; d["month"] = d["Date"].dt.month
        d["quarter"] = d["Date"].dt.quarter; d["doy"] = d["Date"].dt.dayofyear
        d["m_sin"] = np.sin(2*np.pi*d["month"]/12); d["m_cos"] = np.cos(2*np.pi*d["month"]/12)
        d["t"] = np.arange(len(d))
        if "CPI" in d.columns:
            d["lag1"] = d["CPI"].shift(1); d["lag2"] = d["CPI"].shift(2)
            d["rm3"]  = d["CPI"].shift(1).rolling(3).mean()
            d["rs3"]  = d["CPI"].shift(1).rolling(3).std()
        return d

    def fit(self, df):
        self.sarima.fit(df)
        resid = df["CPI"].values - self.sarima.fitted_values
        f = self._feats(df); f["resid"] = resid
        fc = ["year","month","quarter","doy","m_sin","m_cos","t","lag1","lag2","rm3","rs3"]
        cl = f.dropna()
        self.xgb.fit(cl[fc], cl["resid"]); self.feature_cols = fc
        return self

    def predict(self, df_future):
        sp = self.sarima.predict(len(df_future))
        f  = self._feats(df_future).fillna(0)
        corr = self.xgb.predict(f[self.feature_cols])
        return sp + corr


class ProphetForecaster:
    def __init__(self, **kw):
        self.model = Prophet(**kw, interval_width=.95)
        self.fitted = None

    def fit(self, df):
        dp = df.rename(columns={"Date":"ds","CPI":"y"})
        self.fitted = self.model.fit(dp)
        return self

    def predict(self, df_future):
        df_f = df_future.rename(columns={"Date":"ds"})
        fc = self.fitted.predict(df_f)
        return {"forecast": fc["yhat"].values,
                "lower":    fc["yhat_lower"].values,
                "upper":    fc["yhat_upper"].values}


class ProphetBoostForecaster:
    def __init__(self, prophet_params=None, xgb_params=None):
        pp = {"growth":"linear","yearly_seasonality":True,"weekly_seasonality":False,
              "daily_seasonality":False,"n_changepoints":25,"changepoint_range":.8,
              "changepoint_prior_scale":.05,"seasonality_prior_scale":10.}
        xp = {"n_estimators":150,"max_depth":6,"learning_rate":.05,"subsample":.8,
              "colsample_bytree":.8,"random_state":42,"objective":"reg:squarederror",
              "reg_alpha":.1,"reg_lambda":1.}
        self.prophet = ProphetForecaster(**(pp | (prophet_params or {})))
        self.xgb     = XGBRegressor(**(xp | (xgb_params or {})))
        self.feat_cols = None

    def _feats(self, df):
        d = df.copy()
        d["year"] = d["Date"].dt.year; d["month"] = d["Date"].dt.month
        d["quarter"] = d["Date"].dt.quarter; d["doy"] = d["Date"].dt.dayofyear
        d["m_sin"] = np.sin(2*np.pi*d["month"]/12); d["m_cos"] = np.cos(2*np.pi*d["month"]/12)
        d["t"] = np.arange(len(d))
        return d

    def fit(self, df):
        self.prophet.fit(df)
        future = self.prophet.fitted.make_future_dataframe(periods=0, freq="MS")
        pf = self.prophet.fitted.predict(future)["yhat"].values
        resid = df["CPI"].values - pf
        f = self._feats(df); f["resid"] = resid
        fc = ["year","month","quarter","doy","m_sin","m_cos","t"]
        cl = f.dropna()
        self.xgb.fit(cl[fc], cl["resid"]); self.feat_cols = fc
        return self

    def predict(self, df_future):
        pr = self.prophet.predict(df_future)
        f  = self._feats(df_future).fillna(0)
        corr = self.xgb.predict(f[self.feat_cols])
        return {"forecast": pr["forecast"] + corr,
                "lower":    pr["lower"]    + corr,
                "upper":    pr["upper"]    + corr,
                "prophet":  pr["forecast"],
                "xgb_corr": corr}


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def metrics(y_true, y_pred, y_train=None):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = ~(np.isnan(y_true)|np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / np.maximum(yt, 1e-10))) * 100
    if y_train is not None and len(y_train)>1:
        mase = mae / np.mean(np.abs(np.diff(y_train)))
    else:
        mase = mae / (np.mean(np.abs(np.diff(yt))) or 1)
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, MASE=mase)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def chart_historical(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["CPI"],
        mode="lines", name="CPI",
        line=dict(color=CLR["historical"], width=2),
        fill="tozeroy", fillcolor="rgba(100,116,139,.08)"))
    fig.update_layout(**PLOTLY_LAYOUT,
        title="Historical CPI Series", height=380)
    return fig


def chart_decomposition(df, period=12):
    decomp = seasonal_decompose(df.set_index("Date")["CPI"], model="additive", period=period)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Observed","Trend","Seasonal","Residual"],
                        vertical_spacing=.06)
    pairs = [(decomp.observed,"#0f172a"),(decomp.trend,"#6366f1"),
             (decomp.seasonal,"#f59e0b"),(decomp.resid,"#ef4444")]
    for i,(s,c) in enumerate(pairs,1):
        fig.add_trace(go.Scatter(x=s.index,y=s.values,mode="lines",
                                 line=dict(color=c,width=1.5),showlegend=False), row=i, col=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=680,
                      title="Seasonal Decomposition (Additive, Period=12)")
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig


def chart_model_comparison(test_df, preds):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df["Date"], y=test_df["CPI"],
        mode="lines+markers", name="Actual",
        line=dict(color=CLR["actual"], width=3),
        marker=dict(size=5)))
    for key, col in [("sarima",CLR["sarima"]),("boosted_sarima",CLR["boosted_sarima"]),
                     ("prophet",CLR["prophet"]),("prophet_boost",CLR["prophet_boost"])]:
        fig.add_trace(go.Scatter(x=test_df["Date"], y=preds[key],
            mode="lines+markers", name=key.replace("_"," ").title(),
            line=dict(color=col, width=2, dash="dot"),
            marker=dict(size=4)))
    fig.update_layout(**PLOTLY_LAYOUT, height=420,
                      title="All Models — Test Set Predictions vs Actual")
    return fig


def chart_forecast(hist_df, fc_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["CPI"],
        mode="lines", name="Historical",
        line=dict(color=CLR["historical"], width=1.5)))
    fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Upper"],
        mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Lower"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor=CLR["fill"], name="95% CI"))
    fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Forecast"],
        mode="lines+markers", name="Prophet Boost Forecast",
        line=dict(color=CLR["prophet_boost"], width=3),
        marker=dict(size=5)))
    fig.add_vline(x=hist_df["Date"].max(), line_dash="dash",
                  line_color="#e11d48", annotation_text="Forecast start",
                  annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT, height=480,
                      title="36-Month CPI Forecast (Prophet Boost — Winning Model)")
    return fig


def chart_prophet_xgb_split(fc_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=fc_df["Date"], y=fc_df["XGBoost_Correction"],
        name="XGBoost Correction", marker_color="#f59e0b"))
    fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Prophet_Component"],
        mode="lines", name="Prophet Base", line=dict(color="#8b5cf6",width=2.5)))
    fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Forecast"],
        mode="lines", name="Final Forecast", line=dict(color=CLR["prophet_boost"],width=2.5)))
    fig.update_layout(**PLOTLY_LAYOUT, height=420, barmode="relative",
                      title="Forecast Decomposition — Prophet Base + XGBoost Correction")
    return fig


def chart_metrics_radar(comparison_df):
    models = list(comparison_df.index)
    cols   = list(comparison_df.columns)
    norm   = comparison_df.copy()
    for c in cols:
        mn, mx = norm[c].min(), norm[c].max()
        norm[c] = 1 - (norm[c] - mn) / (mx - mn + 1e-10)   # higher = better
    colors = [CLR["sarima"],CLR["boosted_sarima"],CLR["prophet"],CLR["prophet_boost"]]
    fig = go.Figure()
    for mdl, col in zip(models, colors):
        vals = list(norm.loc[mdl]) + [norm.loc[mdl][0]]
        cats = cols + [cols[0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats,
            fill="toself", name=mdl,
            line=dict(color=col), fillcolor=col.replace(")",",0.15)").replace("rgb","rgba") if col.startswith("rgb") else col+"26"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="DM Sans", height=420,
        title="Model Performance Radar (normalised, higher=better)",
        legend=dict(bgcolor="rgba(255,255,255,.8)"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1.5rem;'>
        <div style='font-family:DM Serif Display,serif;font-size:1.3rem;color:#f8fafc;'>
            CPI <em style='color:#38bdf8;font-style:italic;'>Forecaster</em>
        </div>
        <div style='font-size:.72rem;color:#64748b;letter-spacing:.08em;text-transform:uppercase;'>
            Albania · SARIMA · Prophet · XGBoost
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CPI Excel file (.xlsx)", type=["xlsx"])

    st.markdown("<hr style='border-color:#1e293b;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:#64748b;'>Configuration</span>", unsafe_allow_html=True)

    train_ratio   = st.slider("Train ratio", .6, .95, .8, .05,
                               help="Fraction of data used for training")
    forecast_horizon = st.slider("Forecast horizon (months)", 12, 60, 36, 6)
    category_code = st.text_input("Category code", "000000",
                                   help="000000 = Total CPI")

    st.markdown("<hr style='border-color:#1e293b;margin:1rem 0;'>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Full Pipeline", use_container_width=True,
                         type="primary", disabled=(uploaded is None))

    st.markdown("""
    <div style='font-size:.72rem;color:#475569;padding-top:1rem;line-height:1.6;'>
    Based on<br>
    <em style='color:#94a3b8;'>Basha & Gjika (2023) — Forecasting CPI with ARIMA, Prophet and XGBoost</em>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <h1>Albania <em>CPI</em> Forecasting</h1>
    <p>End-to-end time-series forecasting pipeline — comparative analysis of four models</p>
    <span class="badge">SARIMA</span>
    <span class="badge">Boosted SARIMA</span>
    <span class="badge">Prophet</span>
    <span class="badge">Prophet Boost ★</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INITIAL STATE (no file yet)
# ══════════════════════════════════════════════════════════════════════════════

if uploaded is None:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div class="upload-zone">
            <h3>📂 Upload your CPI data</h3>
            <p>Use the sidebar to upload the Albania INSTAT Excel file<br>
            (wide-format with dates in row 3, categories from row 4)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <strong>Pipeline steps after upload:</strong><br>
        1 · Data cleaning & EDA &nbsp;|&nbsp;
        2 · Train / test split &nbsp;|&nbsp;
        3 · Fit 4 models &nbsp;|&nbsp;
        4 · Evaluate metrics &nbsp;|&nbsp;
        5 · 36-month forecast
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PREVIEW DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data(raw_bytes, code):
    df_raw  = pd.read_excel(io.BytesIO(raw_bytes), header=None)
    df_long = clean_wide_to_long(df_raw)
    df_cpi  = get_series(df_long, code)
    return df_long, df_cpi

with st.spinner("Reading file…"):
    raw_bytes = uploaded.read()
    df_long, df_cpi = load_data(raw_bytes, category_code)

# Quick KPIs from the raw data
obs   = len(df_cpi)
cpi_s = df_cpi["CPI"].iloc[0];  cpi_e = df_cpi["CPI"].iloc[-1]
chg   = (cpi_e/cpi_s - 1)*100

st.markdown('<div class="section-heading">Step 1 · <span>Exploratory Data Analysis</span></div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Observations", obs)
k2.metric("Date range", f"{df_cpi['Date'].min().year}–{df_cpi['Date'].max().year}")
k3.metric("CPI start", f"{cpi_s:.1f}")
k4.metric("CPI latest", f"{cpi_e:.1f}", delta=f"{chg:+.1f}% total")

tab_trend, tab_decomp, tab_adf, tab_raw = st.tabs(
    ["📈 Trend", "🔬 Decomposition", "📊 ADF Stationarity", "🗂 Raw data"])

with tab_trend:
    st.plotly_chart(chart_historical(df_cpi), use_container_width=True)

with tab_decomp:
    with st.spinner("Decomposing…"):
        st.plotly_chart(chart_decomposition(df_cpi), use_container_width=True)
    st.markdown("""
    <div class="info-box">
    <strong>Additive decomposition:</strong> Y(t) = Trend(t) + Seasonal(t) + Residual(t) &nbsp;·&nbsp;
    Period = 12 months
    </div>
    """, unsafe_allow_html=True)

with tab_adf:
    adf_res = adfuller(df_cpi["CPI"].dropna())
    is_stat = adf_res[1] <= .05
    c1,c2,c3 = st.columns(3)
    c1.metric("ADF Statistic",  f"{adf_res[0]:.4f}")
    c2.metric("p-value",        f"{adf_res[1]:.4f}")
    c3.metric("Stationarity",   "✅ Stationary" if is_stat else "⚠️ Non-stationary")
    cvdf = pd.DataFrame(adf_res[4].items(), columns=["Level","Critical Value"])
    st.dataframe(cvdf, use_container_width=False)
    if not is_stat:
        st.info("Series is non-stationary → differencing is applied inside SARIMA (d≥1).")

with tab_raw:
    avail_cats = df_long["Category_Code"].unique().tolist()
    chosen = st.selectbox("Category", avail_cats,
                           index=avail_cats.index(category_code) if category_code in avail_cats else 0)
    st.dataframe(df_long[df_long["Category_Code"]==chosen].reset_index(drop=True),
                 use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-heading">Step 2 · <span>Train / Test Split</span></div>',
            unsafe_allow_html=True)

train_df, test_df = chronological_split(df_cpi, train_ratio)

c1, c2, c3 = st.columns(3)
c1.metric("Training obs",  len(train_df),
          f"until {train_df['Date'].max().strftime('%b %Y')}")
c2.metric("Test obs",      len(test_df),
          f"from {test_df['Date'].min().strftime('%b %Y')}")
c3.metric("Split date",    test_df['Date'].min().strftime('%Y-%m-%d'))

fig_split = go.Figure()
fig_split.add_trace(go.Scatter(x=train_df["Date"], y=train_df["CPI"],
    mode="lines", name=f"Train ({int(train_ratio*100)}%)",
    line=dict(color="#0f172a", width=2)))
fig_split.add_trace(go.Scatter(x=test_df["Date"], y=test_df["CPI"],
    mode="lines+markers", name=f"Test ({int((1-train_ratio)*100)}%)",
    line=dict(color="#38bdf8", width=2.5),
    marker=dict(size=5)))
fig_split.add_vline(x=test_df["Date"].min(), line_dash="dash", line_color="#e11d48")
fig_split.update_layout(**PLOTLY_LAYOUT, height=340, title="Chronological 80/20 Split")
st.plotly_chart(fig_split, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FITTING & EVALUATION  (only when Run button is pressed)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-heading">Steps 3–4 · <span>Model Training & Evaluation</span></div>',
            unsafe_allow_html=True)

if not run_btn:
    st.info("👈  Press **Run Full Pipeline** in the sidebar to train all four models.")
    st.stop()

# ── fit with progress ────────────────────────────────────────────────────────

pbar   = st.progress(0, text="Starting pipeline…")
status = st.empty()

preds   = {}
met_all = {}

@st.cache_data(show_spinner=False)
def run_models(train_bytes, test_bytes, tr_ratio, cat):
    tr  = pd.read_json(io.StringIO(train_bytes))
    te  = pd.read_json(io.StringIO(test_bytes))
    tr["Date"] = pd.to_datetime(tr["Date"])
    te["Date"] = pd.to_datetime(te["Date"])
    y_tr = tr["CPI"].values; y_te = te["CPI"].values

    # 1 SARIMA
    m1 = SARIMAForecaster((0,1,0),(1,1,2,12)).fit(tr)
    p1 = m1.predict(len(te))

    # 2 Boosted SARIMA
    m2 = BoostedSARIMA((1,1,1),(0,1,1,12)).fit(tr)
    p2 = m2.predict(te)

    # 3 Prophet
    m3 = ProphetForecaster(growth="linear",yearly_seasonality=True,
                            weekly_seasonality=False,daily_seasonality=False,
                            n_changepoints=25,changepoint_range=.8,
                            changepoint_prior_scale=.05,seasonality_prior_scale=10.)
    m3.fit(tr); p3 = m3.predict(te)["forecast"]

    # 4 Prophet Boost
    m4 = ProphetBoostForecaster().fit(tr)
    r4 = m4.predict(te); p4 = r4["forecast"]

    preds = {"sarima":p1,"boosted_sarima":p2,"prophet":p3,"prophet_boost":p4}
    mets  = {n: metrics(y_te, pv, y_tr) for n,pv in preds.items()}
    return preds, mets, m4

# serialise DataFrames for cache key
train_bytes = train_df.to_json()
test_bytes  = test_df.to_json()

pbar.progress(5, "Fitting SARIMA (1/4)…")
preds, met_all, pb_model_test = run_models(train_bytes, test_bytes, train_ratio, category_code)
pbar.progress(100, "All models fitted ✓")
status.empty(); pbar.empty()

# ── metrics table ─────────────────────────────────────────────────────────────

cmp = pd.DataFrame(met_all).T.round(4)
best_model = cmp["RMSE"].idxmin()

st.markdown(f"""
<p>
Best model by RMSE: 
<strong>{best_model.replace('_',' ').title()}</strong>
<span class="winner-badge">★ WINNER</span>
</p>
""", unsafe_allow_html=True)

col_table, col_radar = st.columns([1,1])

with col_table:
    def style_cmp(df):
        s = df.style
        for col in df.columns:
            min_idx = df[col].idxmin()
            s = s.highlight_min(subset=[col], color="#dcfce7")
        return s.format("{:.4f}")
    st.dataframe(style_cmp(cmp), use_container_width=True, height=220)

    st.markdown("""
    <div class="info-box" style="font-size:.8rem;">
    <strong>Metrics:</strong> RMSE · MAE · MAPE (%) · MASE &nbsp;—&nbsp;
    green cells = best per metric
    </div>
    """, unsafe_allow_html=True)

with col_radar:
    st.plotly_chart(chart_metrics_radar(cmp), use_container_width=True)

# ── model comparison chart ────────────────────────────────────────────────────

st.plotly_chart(chart_model_comparison(test_df, preds), use_container_width=True)

# ── per-model detail tabs ─────────────────────────────────────────────────────

t1, t2, t3, t4 = st.tabs(["SARIMA","Boosted SARIMA","Prophet","Prophet Boost ★"])

def residual_chart(y_true, y_pred, name, color):
    resid = np.array(y_true) - np.array(y_pred)
    fig   = make_subplots(rows=1, cols=2,
                          subplot_titles=["Residuals over time","Residual distribution"])
    fig.add_trace(go.Scatter(x=list(range(len(resid))), y=resid,
        mode="lines+markers", marker=dict(size=4,color=color),
        line=dict(color=color,width=1.5), name="Residual"), row=1,col=1)
    fig.add_trace(go.Histogram(x=resid, marker_color=color,
        opacity=.7, name="Freq"), row=1,col=2)
    fig.update_layout(**PLOTLY_LAYOUT, height=300, title=f"{name} — Residual Analysis",
                      showlegend=False)
    return fig

for tab, key, col in zip([t1,t2,t3,t4],
                          ["sarima","boosted_sarima","prophet","prophet_boost"],
                          [CLR["sarima"],CLR["boosted_sarima"],CLR["prophet"],CLR["prophet_boost"]]):
    with tab:
        m = met_all[key]
        mc1,mc2,mc3,mc4 = tab.columns(4)
        mc1.metric("RMSE",  f"{m['RMSE']:.4f}")
        mc2.metric("MAE",   f"{m['MAE']:.4f}")
        mc3.metric("MAPE",  f"{m['MAPE']:.2f}%")
        mc4.metric("MASE",  f"{m['MASE']:.4f}")
        tab.plotly_chart(residual_chart(test_df["CPI"], preds[key],
                         key.replace("_"," ").title(), col), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 36-MONTH FORECAST  (Prophet Boost on full dataset)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f'<div class="section-heading">Step 5 · <span>{forecast_horizon}-Month Future Forecast</span></div>',
            unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def run_final_forecast(cpi_bytes, horizon):
    df = pd.read_json(io.StringIO(cpi_bytes))
    df["Date"] = pd.to_datetime(df["Date"])
    model = ProphetBoostForecaster().fit(df)
    future = pd.DataFrame({"Date": pd.date_range(
        start=df["Date"].max() + pd.DateOffset(months=1),
        periods=horizon, freq="MS")})
    fc = model.predict(future)
    return pd.DataFrame({
        "Date":               future["Date"],
        "Forecast":           fc["forecast"],
        "Lower":              fc["lower"],
        "Upper":              fc["upper"],
        "Prophet_Component":  fc["prophet"],
        "XGBoost_Correction": fc["xgb_corr"],
    })

with st.spinner(f"Fitting Prophet Boost on full dataset and forecasting {forecast_horizon} months…"):
    fc_df = run_final_forecast(df_cpi.to_json(), forecast_horizon)

# summary KPIs
last_obs = df_cpi["CPI"].iloc[-1]
fc_end   = fc_df["Forecast"].iloc[-1]
fc_chg   = (fc_end/last_obs - 1)*100

k1,k2,k3,k4 = st.columns(4)
k1.metric("Last observed CPI",  f"{last_obs:.2f}")
k2.metric("Forecast end CPI",   f"{fc_end:.2f}", delta=f"{fc_end-last_obs:+.2f}")
k3.metric("Expected growth",    f"{fc_chg:+.2f}%")
k4.metric("Forecast horizon",   f"{forecast_horizon} months")

st.plotly_chart(chart_forecast(df_cpi, fc_df), use_container_width=True)
st.plotly_chart(chart_prophet_xgb_split(fc_df), use_container_width=True)

# forecast table
st.markdown("#### 📋 Forecast Table")
fc_display = fc_df.copy()
fc_display["Date"] = fc_display["Date"].dt.strftime("%Y-%m")
for c in ["Forecast","Lower","Upper","Prophet_Component","XGBoost_Correction"]:
    fc_display[c] = fc_display[c].round(2)
st.dataframe(fc_display, use_container_width=True, height=400)

# ── download button ───────────────────────────────────────────────────────────
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    cmp.to_excel(w, sheet_name="Model_Comparison")
    fc_df.to_excel(w, sheet_name="Forecast", index=False)
    df_cpi.to_excel(w, sheet_name="Historical", index=False)
    pd.DataFrame({
        "Date":          test_df["Date"],
        "Actual":        test_df["CPI"].values,
        "SARIMA":        preds["sarima"],
        "BoostedSARIMA": preds["boosted_sarima"],
        "Prophet":       preds["prophet"],
        "ProphetBoost":  preds["prophet_boost"],
    }).to_excel(w, sheet_name="Test_Predictions", index=False)

st.download_button(
    "⬇  Download all results (.xlsx)",
    data=buf.getvalue(),
    file_name="albania_cpi_forecast_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;font-size:.78rem;color:#94a3b8;'>
Albania CPI Forecasting Pipeline &nbsp;·&nbsp; 
Based on Basha & Gjika (2023) &nbsp;·&nbsp;
SARIMA · Prophet · XGBoost hybrid models
</div>
""", unsafe_allow_html=True)
