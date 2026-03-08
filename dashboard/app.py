"""
=============================================================================
DASHBOARD: GX×DX Closed-Loop Digital Twin — Interactive Streamlit App
=============================================================================
Project : GX×DX Closed-Loop Digital Twin
Author  : V Niranjana | IIT Jodhpur
Design  : Industrial Control Room — SCADA-inspired precision aesthetic
=============================================================================
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.datacenter_thermal import DataCenterThermalModel, get_cpu_profile
from core.heat_exchanger      import HeatExchanger
from core.mea_regenerator     import MEARegenerator
from core.solar_battery       import SolarBatterySystem, fetch_nasa_solar, get_typical_day
from core.optimizer           import SystemOptimizer

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title  = "GX×DX Digital Twin",
    page_icon   = "⚡",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ============================================================================
# LOCATION PRESETS
# ============================================================================

LOCATIONS = {
    "🇮🇳  Jodhpur, India (Default)"        : {"lat": 26.9,  "lon": 73.0,  "tz": "Asia/Kolkata",    "grid_ef": 0.82, "temps": [28,27,26,26,27,29,32,35,38,40,41,42,42,41,40,39,38,36,34,32,31,30,29,28]},
    "🇮🇳  New Delhi, India"                : {"lat": 28.6,  "lon": 77.2,  "tz": "Asia/Kolkata",    "grid_ef": 0.82, "temps": [25,24,23,23,24,26,29,33,37,39,40,41,41,40,39,37,36,34,31,29,28,27,26,25]},
    "🇯🇵  Tokyo, Japan"                    : {"lat": 35.7,  "lon": 139.7, "tz": "Asia/Tokyo",      "grid_ef": 0.47, "temps": [22,21,21,20,21,22,25,28,31,33,34,35,35,34,33,31,30,28,26,25,24,23,22,22]},
    "🇯🇵  Osaka, Japan"                    : {"lat": 34.7,  "lon": 135.5, "tz": "Asia/Tokyo",      "grid_ef": 0.47, "temps": [23,22,21,21,22,23,26,29,32,34,35,36,36,35,34,32,31,29,27,26,25,24,23,23]},
    "🇯🇵  Fukuoka, Japan"                  : {"lat": 33.6,  "lon": 130.4, "tz": "Asia/Tokyo",      "grid_ef": 0.47, "temps": [24,23,22,22,23,24,27,30,33,35,36,37,37,36,35,33,32,30,28,27,26,25,24,24]},
    "🇩🇪  Munich, Germany"                 : {"lat": 48.1,  "lon": 11.6,  "tz": "Europe/Berlin",   "grid_ef": 0.38, "temps": [15,14,14,13,14,16,19,22,25,27,28,28,28,27,26,24,23,21,19,18,17,16,15,15]},
    "🇸🇦  Riyadh, Saudi Arabia"            : {"lat": 24.7,  "lon": 46.7,  "tz": "Asia/Riyadh",     "grid_ef": 0.75, "temps": [32,31,30,30,31,33,36,39,42,44,45,46,46,45,44,42,40,38,36,34,33,32,31,31]},
    "🇺🇸  Phoenix, USA"                    : {"lat": 33.4,  "lon": -112.1,"tz": "America/Phoenix",  "grid_ef": 0.45, "temps": [30,29,28,28,29,31,34,37,40,42,43,44,44,43,42,40,38,36,34,33,32,31,30,30]},
    "🇸🇬  Singapore"                       : {"lat": 1.3,   "lon": 103.8, "tz": "Asia/Singapore",  "grid_ef": 0.43, "temps": [27,27,26,26,27,27,28,29,30,31,32,32,32,32,31,30,30,29,28,28,28,27,27,27]},
    "✏️   Custom Coordinates"              : {"lat": None,  "lon": None,  "tz": "UTC",             "grid_ef": 0.60, "temps": [25,24,23,23,24,26,29,32,35,37,38,39,39,38,37,35,33,31,29,28,27,26,25,25]},
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ============================================================================
# CUSTOM CSS — Industrial Control Room Aesthetic
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-deep:     #020810;
    --bg-panel:    #040d1a;
    --bg-card:     #071428;
    --bg-elevated: #0a1e3d;
    --accent-cyan:  #00d4ff;
    --accent-green: #00ff9d;
    --accent-amber: #ffb700;
    --accent-red:   #ff3d5a;
    --accent-violet:#7c3aed;
    --text-primary: #e2f4ff;
    --text-muted:   #5a8aaa;
    --text-dim:     #2a4a62;
    --border:       rgba(0,212,255,0.12);
    --border-bright:rgba(0,212,255,0.35);
    --glow-cyan:    0 0 20px rgba(0,212,255,0.25);
    --glow-green:   0 0 20px rgba(0,255,157,0.2);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--text-primary);
}

/* ── App Background with animated mesh ── */
.stApp {
    background:
        radial-gradient(ellipse at 20% 20%, rgba(0,100,200,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(0,200,150,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(0,50,120,0.05) 0%, transparent 70%),
        var(--bg-deep);
    min-height: 100vh;
}

/* ── Animated grid overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ── Main content above grid ── */
.main .block-container {
    position: relative;
    z-index: 1;
    padding-top: 1rem;
    max-width: 1600px;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg,
        rgba(0,212,255,0.08) 0%,
        rgba(0,100,180,0.06) 40%,
        rgba(0,255,157,0.04) 100%
    );
    border: 1px solid var(--border-bright);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 200px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,255,157,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent-green);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 8px;
    opacity: 0.9;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0 0 6px 0;
    line-height: 1.1;
}
.hero-title span {
    color: var(--accent-cyan);
}
.hero-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-muted);
    font-size: 0.88rem;
    font-weight: 300;
    letter-spacing: 0.3px;
    margin-top: 8px;
}
.hero-tags {
    display: flex;
    gap: 8px;
    margin-top: 14px;
    flex-wrap: wrap;
}
.hero-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent-cyan);
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 4px;
    padding: 3px 10px;
    letter-spacing: 1px;
}
.hero-location {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent-amber);
    margin-top: 10px;
    opacity: 0.9;
}

/* ── KPI Cards ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease;
}
.kpi-card:hover {
    border-color: var(--border-bright);
    box-shadow: var(--glow-cyan);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-line, var(--accent-cyan));
    opacity: 0.7;
}
.kpi-icon {
    font-size: 1.1rem;
    margin-bottom: 8px;
    display: block;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 500;
}
.kpi-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    margin-top: 3px;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin: 20px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-bright), transparent);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020d1f 0%, #030810 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif !important;
    color: var(--accent-cyan) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}
section[data-testid="stSidebar"] label {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--accent-cyan) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-muted) !important;
    border-radius: 7px !important;
    padding: 8px 16px !important;
    letter-spacing: 0.5px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,212,255,0.08)) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 16px !important;
}

/* ── Metrics override ── */
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}
div[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.4rem !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: var(--accent-green) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* ── Selectbox & Sliders ── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] * {
    background-color: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--accent-cyan) !important;
}

/* ── Info boxes ── */
.stInfo {
    background: rgba(0,212,255,0.06) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent-green);
    background: rgba(0,255,157,0.08);
    border: 1px solid rgba(0,255,157,0.2);
    border-radius: 20px;
    padding: 4px 12px;
}
.status-dot {
    width: 6px; height: 6px;
    background: var(--accent-green);
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── Pipeline flow ── */
.pipeline-container {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 16px 0;
    overflow-x: auto;
    padding: 4px 0;
}
.pipeline-node {
    background: var(--bg-elevated);
    border: 1px solid var(--border-bright);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    min-width: 120px;
    flex-shrink: 0;
}
.pipeline-node-icon { font-size: 1.4rem; display: block; margin-bottom: 4px; }
.pipeline-node-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.pipeline-node-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-primary);
    margin-top: 3px;
    font-weight: 500;
}
.pipeline-arrow {
    color: var(--accent-cyan);
    font-size: 1.2rem;
    opacity: 0.5;
    padding: 0 4px;
    flex-shrink: 0;
}

/* ── Comparison table ── */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
}
.compare-table th {
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.68rem;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-bright);
    text-align: left;
}
.compare-table td {
    color: var(--text-primary);
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
}
.compare-table tr:hover td {
    background: rgba(0,212,255,0.04);
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 24px;
    margin-top: 32px;
    border-top: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 1px;
}

/* ── Hide streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PLOT THEME
# ============================================================================

PLOT_BG = "rgba(4,13,26,0.95)"
PLOT_PANEL = "rgba(7,20,40,0.8)"

PLOT_LAYOUT = dict(
    paper_bgcolor = PLOT_BG,
    plot_bgcolor  = PLOT_PANEL,
    font          = dict(family="IBM Plex Mono", color="#5a8aaa", size=11),
    xaxis         = dict(
        gridcolor="#0a2040", gridwidth=1,
        showline=True, linecolor="rgba(0,212,255,0.2)",
        tickfont=dict(color="#5a8aaa", size=10),
        zeroline=False
    ),
    yaxis         = dict(
        gridcolor="#0a2040", gridwidth=1,
        showline=True, linecolor="rgba(0,212,255,0.2)",
        tickfont=dict(color="#5a8aaa", size=10),
        zeroline=False
    ),
    legend        = dict(
        bgcolor="rgba(4,13,26,0.9)",
        bordercolor="rgba(0,212,255,0.2)",
        borderwidth=1,
        font=dict(color="#5a8aaa", size=10),
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="left", x=0
    ),
    margin        = dict(l=48, r=16, t=98, b=40),
    hoverlabel    = dict(
        bgcolor="#040d1a",
        bordercolor="rgba(0,212,255,0.4)",
        font=dict(family="IBM Plex Mono", color="#e2f4ff", size=11)
    ),
)

COLORS = {
    "solar"    : "#ffb700",
    "thermal"  : "#ff6b35",
    "waste"    : "#00ff9d",
    "ccus"     : "#00d4ff",
    "grid"     : "#ff3d5a",
    "battery"  : "#a855f7",
    "co2"      : "#38bdf8",
    "credits"  : "#4ade80",
    "pv"       : "#fde047",
}

HOURS       = list(range(24))
HOUR_LABELS = [f"{h:02d}:00" for h in HOURS]

# ============================================================================
# DATA & MODEL LOADING
# ============================================================================

@st.cache_resource
def load_base_models():
    dc    = DataCenterThermalModel(it_capacity_kw=100, pue=1.4)
    hx    = HeatExchanger(area_m2=25.0, glycol_flow_kgs=0.8, glycol_inlet_temp_c=25.0)
    mea   = MEARegenerator(mea_flow_kgs=0.21, target_capture_rate=0.90)
    solar = SolarBatterySystem(pv_capacity_kwp=150.0, thermal_area_m2=300.0, battery_capacity_kwh=200.0)
    return dc, hx, mea, solar

@st.cache_data(ttl=3600)
def load_solar_data(lat, lon, location_name):
    """Load NASA solar data — cached per location."""
    cache_dir  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(cache_dir, exist_ok=True)
    safe_name  = location_name.replace(" ", "_").replace(",", "").replace("/", "_")[:30]
    cache_path = os.path.join(cache_dir, f"solar_{safe_name}_{lat}_{lon}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["ghi_wm2"]

    return fetch_nasa_solar(lat=lat, lon=lon, year=2023)

@st.cache_data
def run_simulation(
    cpu_profile_name, thermal_area, pv_capacity,
    battery_capacity, month, pue, mea_flow,
    lat, lon, location_key
):
    dc, hx, mea, solar = load_base_models()
    dc.pue                     = pue
    mea.mea_flow_kgs           = mea_flow
    solar.thermal_area_m2      = thermal_area
    solar.pv_capacity_kwp      = pv_capacity
    solar.pv_area_m2           = pv_capacity / 0.20
    solar.battery_capacity_kwh = battery_capacity

    loc_data    = LOCATIONS[location_key]
    ghi_annual  = load_solar_data(lat, lon, location_key)
    ghi_profile = get_typical_day(ghi_annual, month=month)
    cpu_profile = get_cpu_profile(cpu_profile_name)
    amb_temps   = loc_data["temps"]

    opt    = SystemOptimizer(dc, hx, mea, solar)
    result = opt.simulate_system(cpu_profile, ghi_profile, ambient_temps=amb_temps)
    return result, ghi_profile, cpu_profile

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    st.sidebar.markdown("""
    <div style='padding:16px 0 8px 0;'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;
                    color:#00d4ff;letter-spacing:1px;'>⚡ SYSTEM CONFIG</div>
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;
                    color:#2a4a62;margin-top:4px;letter-spacing:2px;'>
            GX×DX DIGITAL TWIN v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # ── Location ──
    st.sidebar.markdown("### 📍 Location")
    location_key = st.sidebar.selectbox(
        "Select Location",
        options=list(LOCATIONS.keys()),
        index=0,
        help="Choose a preset city or enter custom coordinates"
    )

    loc_data = LOCATIONS[location_key]
    if loc_data["lat"] is None:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            lat = st.sidebar.number_input("Latitude",  value=26.9, min_value=-90.0,  max_value=90.0,  step=0.1, format="%.2f")
        with col2:
            lon = st.sidebar.number_input("Longitude", value=73.0, min_value=-180.0, max_value=180.0, step=0.1, format="%.2f")
        grid_ef = st.sidebar.number_input("Grid Emission Factor (kgCO₂/kWh)", value=0.60, min_value=0.1, max_value=1.2, step=0.01, format="%.2f")
    else:
        lat     = loc_data["lat"]
        lon     = loc_data["lon"]
        grid_ef = loc_data["grid_ef"]
        st.sidebar.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;"
            f"color:#ffb700;padding:6px 0;'>📌 {lat}°N, {lon}°E | Grid EF: {grid_ef} kgCO₂/kWh</div>",
            unsafe_allow_html=True
        )

    month = st.sidebar.selectbox(
        "Simulation Month",
        options=list(range(1, 13)),
        format_func=lambda m: MONTH_NAMES[m-1],
        index=5
    )

    st.sidebar.markdown("---")

    # ── Data Center ──
    st.sidebar.markdown("### 🖥️ Data Center")
    pue = st.sidebar.slider("PUE (Power Usage Effectiveness)",
        min_value=1.1, max_value=2.0, value=1.4, step=0.05,
        help="1.0 = perfect. Modular DC typical: 1.3–1.5")
    cpu_profile = st.sidebar.selectbox(
        "Server Workload Profile",
        ["office", "cloud", "night_batch"],
        format_func=lambda x: {"office":"🏢 Office Hours","cloud":"☁️ Cloud Always-On","night_batch":"🌙 Night Batch"}[x]
    )

    st.sidebar.markdown("---")

    # ── Solar System ──
    st.sidebar.markdown("### ☀️ Renewable System")
    thermal_area = st.sidebar.slider("Solar Thermal Area (m²)",
        min_value=100, max_value=1500, value=300, step=50,
        help="Flat-plate collectors feeding MEA regenerator")
    pv_capacity = st.sidebar.slider("Solar PV Capacity (kWp)",
        min_value=50, max_value=300, value=150, step=10,
        help="Photovoltaic array powering data center")
    battery_capacity = st.sidebar.slider("Battery Storage (kWh)",
        min_value=50, max_value=500, value=200, step=25,
        help="Li-ion buffer for overnight DC operation")

    st.sidebar.markdown("---")

    # ── CCUS ──
    st.sidebar.markdown("### 🏭 CCUS Unit")
    mea_flow = st.sidebar.slider("MEA Flow Rate (kg/s)",
        min_value=0.10, max_value=0.50, value=0.21, step=0.01,
        help="30 wt% MEA solution through regenerator")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
        "color:#2a4a62;text-align:center;line-height:1.8;'>"
        "V NIRANJANA · IIT JODHPUR<br>"
        "B.Tech Chemical Engineering<br>"
        "GX×DX DIGITAL TWIN · 2025"
        "</div>",
        unsafe_allow_html=True
    )

    return (pue, cpu_profile, thermal_area, pv_capacity,
            battery_capacity, month, mea_flow, lat, lon, location_key, grid_ef)

# ============================================================================
# HERO BANNER
# ============================================================================

def render_hero(location_key, lat, lon, month):
    loc_name = location_key.split("  ")[-1] if "  " in location_key else location_key
    month_name = MONTH_NAMES[month-1]

    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-eyebrow">▸ LIVE SIMULATION RUNNING</div>
        <div class="hero-title">GX<span>×</span>DX Closed-Loop <span>Digital Twin</span></div>
        <div class="hero-subtitle">
            Waste Heat Recovery · Modular Data Center → MEA-CCUS Integration ·
            Renewable Energy Optimization · Carbon Credit Quantification
        </div>
        <div class="hero-tags">
            <span class="hero-tag">CCUS</span>
            <span class="hero-tag">HEAT RECOVERY</span>
            <span class="hero-tag">SOLAR PV + THERMAL</span>
            <span class="hero-tag">BATTERY STORAGE</span>
            <span class="hero-tag">MEA SCRUBBING</span>
            <span class="hero-tag">CARBON CREDITS</span>
        </div>
        <div class="hero-location">
            📍 {loc_name} &nbsp;·&nbsp; {lat}°N {lon}°E &nbsp;·&nbsp;
            {month_name} Irradiance Profile &nbsp;·&nbsp; 100 kW Modular DC
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# KPI CARDS
# ============================================================================

def render_kpis(summary, thermal_area, pv_capacity, battery_capacity):
    cards = [
        ("🌿", f"{summary['total_co2_captured_kg']:,.0f}", "CO₂ Captured", "kg / day",     "#00ff9d"),
        ("💰", f"${summary['total_carbon_credits_usd']:.0f}", "Carbon Credits", "USD / day","#4ade80"),
        ("☀️", f"{summary['total_solar_thermal_kwh']:.0f}",  "Solar Thermal",  "kWh / day", "#ffb700"),
        ("♨️", f"{summary['total_waste_heat_kwh']:.0f}",     "Waste Heat Used","kWh / day", "#ff6b35"),
        ("🔋", f"{summary['total_grid_import_kwh']:.0f}",    "Grid Import",    "kWh / day", "#a855f7"),
        ("🌍", f"{summary['avg_combined_re_pct']:.1f}%",     "RE Fraction",    "combined",  "#00d4ff"),
    ]

    cols = st.columns(6)
    for col, (icon, value, label, unit, color) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent-line:{color};">
                <span class="kpi-icon">{icon}</span>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-unit">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PIPELINE FLOW
# ============================================================================

def render_pipeline(hourly, summary):
    st.markdown('<div class="section-header">System Pipeline — Peak Hour Energy Flow</div>',
                unsafe_allow_html=True)

    peak = hourly[12]  # noon

    nodes = [
        ("☀️", "Solar PV",     f"{peak['P_pv_kw']:.1f} kW"),
        ("🔋", "Battery",      f"{peak['soc_pct']:.0f}% SOC"),
        ("🖥️", "Data Center",  f"{peak['dc_power_kw']:.1f} kW"),
        ("♨️", "Heat Exchanger",f"{peak['hx_Q_kw']:.1f} kW"),
        ("🌡️", "Solar Thermal", f"{peak['Q_thermal_kw']:.1f} kW"),
        ("🏭", "MEA Regen.",   f"{peak['Q_total_required_kw']:.1f} kW"),
        ("✅", "CO₂ Captured", f"{peak['co2_captured_kgh']:.1f} kg/h"),
    ]

    html = '<div class="pipeline-container">'
    for i, (icon, label, val) in enumerate(nodes):
        html += f"""
        <div class="pipeline-node">
            <span class="pipeline-node-icon">{icon}</span>
            <div class="pipeline-node-label">{label}</div>
            <div class="pipeline-node-value">{val}</div>
        </div>"""
        if i < len(nodes) - 1:
            html += '<div class="pipeline-arrow">→</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# SANKEY DIAGRAM
# ============================================================================

def render_sankey(hourly):
    st.markdown('<div class="section-header">Energy Flow — Sankey Diagram (Peak Hour)</div>',
                unsafe_allow_html=True)

    peak = hourly[12]
    pv   = max(peak["P_pv_kw"], 0.5)
    batt = max(peak.get("discharge_kw", 1.0), 0.5)
    grid = max(peak["P_grid_kw"], 0.5)
    dc   = max(peak["dc_power_kw"], 0.5)
    wh   = max(peak["Q_waste_heat_kw"], 0.5)
    sol  = max(peak["Q_solar_to_ccus_kw"], 0.5)
    req  = max(peak["Q_total_required_kw"], 0.5)
    co2  = max(peak["co2_captured_kgh"] / 10, 0.5)
    def_ = max(peak["Q_ccus_deficit_kw"], 0.5)

    fig = go.Figure(go.Sankey(
        arrangement = "fixed",
        node = dict(
            pad=25, thickness=22,
            line=dict(color="rgba(0,212,255,0.3)", width=0.5),
            label=["☀️ Solar PV","🔋 Battery","🌐 Grid","🖥️ Data Center",
                   "♨️ Heat Exchanger","🌡️ Solar Thermal","🏭 MEA Regen",
                   "✅ CO₂ Captured","⚡ Grid Thermal"],
            color=["#b45309","#7c3aed","#dc2626","#0369a1",
                   "#065f46","#92400e","#0e7490","#15803d","#dc2626"],
            x=[0.0, 0.15, 0.0, 0.32, 0.55, 0.32, 0.72, 1.0, 0.55],
            y=[0.1,  0.35, 0.65, 0.3, 0.45, 0.75, 0.45, 0.45, 0.85],
        ),
        link = dict(
            source=[0,  1,  2,  3,  4,  5,  6,  2  ],
            target=[3,  3,  3,  4,  6,  6,  7,  8  ],
            value =[pv, batt, grid, wh, wh, sol, co2, def_],
            color =[
                "rgba(251,191,36,0.35)",
                "rgba(167,139,250,0.35)",
                "rgba(248,113,113,0.35)",
                "rgba(52,211,153,0.35)",
                "rgba(52,211,153,0.35)",
                "rgba(251,146,60,0.35)",
                "rgba(56,189,248,0.35)",
                "rgba(248,113,113,0.25)",
            ]
        )
    ))
    fig.update_layout(
        title=dict(text="Peak Hour (12:00) — Energy Flows [kW]",
                   font=dict(color="#5a8aaa", size=12, family="IBM Plex Mono")),
        height=360, **PLOT_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TABS
# ============================================================================

def render_tabs(hourly, summary, ghi_profile, cpu_profile, location_key, thermal_area, pv_capacity, battery_capacity):

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🖥️  Data Center",
        "☀️  Solar & Battery",
        "🏭  CCUS",
        "💰  Business Case",
        "🌍  Location Compare",
        "📊  Full Overview",
    ])

    df = pd.DataFrame(hourly)
    df["hour_label"] = HOUR_LABELS

    # ────────────────────────────────────────────────────────────
    # TAB 1: DATA CENTER
    # ────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Data Center Thermal Profile</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=HOUR_LABELS, y=df["dc_power_kw"],
                name="Facility Power", marker_color=COLORS["ccus"],
                marker_line_width=0, opacity=0.75
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=HOUR_LABELS, y=df["dc_waste_heat_kw"],
                name="Waste Heat", line=dict(color=COLORS["thermal"], width=2.5),
                mode="lines+markers", marker=dict(size=4, color=COLORS["thermal"])
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=HOUR_LABELS, y=df["cpu_pct"],
                name="CPU %", line=dict(color=COLORS["solar"], width=2, dash="dot"),
            ), secondary_y=True)
            fig.update_layout(title="Power Consumption & Waste Heat",height=320,**PLOT_LAYOUT)
            fig.update_yaxes(title_text="Power [kW]", secondary_y=False, color="#5a8aaa")
            fig.update_yaxes(title_text="CPU [%]",    secondary_y=True,  color=COLORS["solar"])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=HOUR_LABELS, y=df["glycol_out_c"],
                fill="tozeroy", fillcolor="rgba(0,255,157,0.07)",
                line=dict(color=COLORS["waste"], width=2.5),
                name="Glycol Outlet Temp"
            ))
            fig.add_hrect(y0=0, y1=40, fillcolor="rgba(0,212,255,0.04)",
                          line_width=0, annotation_text="Pre-heat Zone",
                          annotation_font_color="#2a4a62")
            fig.add_hline(y=120, line_dash="dash", line_color=COLORS["grid"], line_width=1.5,
                          annotation_text="MEA Regen Target 120°C",
                          annotation_font=dict(color=COLORS["grid"], size=10))
            fig.add_hline(y=25, line_dash="dot", line_color="#2a4a62", line_width=1,
                          annotation_text="Glycol Inlet 25°C",
                          annotation_font=dict(color="#2a4a62", size=10))
            fig.update_layout(title="Glycol Loop Temperature Profile",
                              yaxis_title="Temperature [°C]", height=320, **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Heat Recovery Metrics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        total_wh_gen  = sum(df["dc_waste_heat_kw"])
        total_wh_used = summary["total_waste_heat_kwh"]
        c1.metric("Waste Heat Generated",  f"{total_wh_gen:.0f} kWh/day")
        c2.metric("Waste Heat Captured",   f"{total_wh_used:.0f} kWh/day")
        c3.metric("Heat Recovery Rate",    f"{total_wh_used/total_wh_gen*100:.1f}%")
        c4.metric("Avg Rack Outlet Temp",  f"{df['glycol_out_c'].mean():.1f} °C")

        # Heat recovery gap explanation
        st.info(
            f"💡 **Engineering Insight:** The glycol loop delivers heat at ~34°C, but MEA regeneration "
            f"requires 120°C. This **86°C temperature gap** is the core engineering challenge — "
            f"solar thermal collectors bridge this deficit, and the optimizer finds the minimum "
            f"collector area to hit your target renewable coverage."
        )

    # ────────────────────────────────────────────────────────────
    # TAB 2: SOLAR & BATTERY
    # ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Solar Generation Profile</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=HOUR_LABELS, y=df["P_pv_kw"],
                name="Solar PV [kW]", marker_color=COLORS["pv"], opacity=0.85, marker_line_width=0
            ), secondary_y=False)
            fig.add_trace(go.Bar(
                x=HOUR_LABELS, y=df["Q_thermal_kw"],
                name="Solar Thermal [kW]", marker_color=COLORS["thermal"], opacity=0.85, marker_line_width=0
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=HOUR_LABELS, y=ghi_profile,
                name="GHI [W/m²]", line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dot")
            ), secondary_y=True)

            # --- DEBUG FIX START ---
            # We create a copy and remove 'legend' and 'margin' to avoid the "multiple values" error
            clean_layout = {k: v for k, v in PLOT_LAYOUT.items() if k not in ['legend', 'margin']}
            
            fig.update_layout(
                title="PV + Thermal Generation vs Irradiance",
                barmode="group", 
                height=320, 
                margin=dict(t=90),  # Room for the legend
                legend=dict(
                    y=1.3,          # Pushed higher
                    x=0.5, 
                    xanchor="center",
                    orientation="h"
                ),
                **clean_layout      # Unpack only the safe keys
            )
            # --- DEBUG FIX END ---

            fig.update_yaxes(title_text="Generation [kW]", secondary_y=False, color="#5a8aaa")
            fig.update_yaxes(title_text="GHI [W/m²]",      secondary_y=True,  color="rgba(255,255,255,0.5)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=HOUR_LABELS, y=df["soc_pct"],
                fill="tozeroy", fillcolor="rgba(168,85,247,0.12)",
                line=dict(color=COLORS["battery"], width=2.5),
                name="Battery SOC [%]"
            ), secondary_y=False)
            fig.add_trace(go.Bar(
                x=HOUR_LABELS, y=df["P_grid_kw"],
                name="Grid Import [kW]", marker_color=COLORS["grid"],
                opacity=0.6, marker_line_width=0
            ), secondary_y=True)
            fig.add_hline(y=10, line_dash="dot", line_color="#2a4a62", line_width=1,
                          annotation_text="SOC Min 10%",
                          annotation_font=dict(color="#2a4a62", size=9))
            fig.update_layout(title="Battery State of Charge & Grid Import",
                              height=320, **PLOT_LAYOUT)
            fig.update_yaxes(title_text="SOC [%]",          secondary_y=False, color=COLORS["battery"])
            fig.update_yaxes(title_text="Grid Import [kW]", secondary_y=True,  color=COLORS["grid"])
            st.plotly_chart(fig, use_container_width=True)

        # RE fraction
        st.markdown('<div class="section-header">Renewable Energy Fraction</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=HOUR_LABELS, y=df["combined_re_pct"],
            fill="tozeroy", fillcolor="rgba(74,222,128,0.08)",
            line=dict(color=COLORS["credits"], width=2.5),
            name="Combined RE %",
            hovertemplate="%{x}<br>RE Fraction: %{y:.1f}%<extra></extra>"
        ))
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(74,222,128,0.05)",
                      line_width=0, annotation_text="Target Zone ≥ 80%",
                      annotation_font=dict(color="#4ade80", size=10))
        fig.add_hline(y=80, line_dash="dash", line_color=COLORS["credits"], line_width=1.5,
                      annotation_text="80% RE Target",
                      annotation_font=dict(color=COLORS["credits"], size=10))
        fig.update_layout(
            title="Hourly Combined Renewable Energy Fraction (Electricity + Thermal)",
            yaxis_title="RE Fraction [%]", yaxis_range=[0, 105], height=280, **PLOT_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total PV Generation",   f"{summary['total_pv_kwh']:.0f} kWh/day")
        c2.metric("Total Solar Thermal",   f"{summary['total_solar_thermal_kwh']:.0f} kWh/day")
        c3.metric("Avg RE Fraction",       f"{summary['avg_combined_re_pct']:.1f}%")
        c4.metric("Peak PV Output",        f"{df['P_pv_kw'].max():.1f} kW")

    # ────────────────────────────────────────────────────────────
    # TAB 3: CCUS
    # ────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">CCUS Heat Supply Stack</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=HOUR_LABELS, y=df["Q_waste_heat_kw"],
                name="♨️ Waste Heat", marker_color=COLORS["waste"], marker_line_width=0
            ))
            fig.add_trace(go.Bar(
                x=HOUR_LABELS, y=df["Q_solar_to_ccus_kw"],
                name="☀️ Solar Thermal", marker_color=COLORS["solar"], marker_line_width=0
            ))
            fig.add_trace(go.Bar(
                x=HOUR_LABELS, y=df["Q_ccus_deficit_kw"],
                name="🔌 Grid Backup", marker_color=COLORS["grid"],
                opacity=0.55, marker_line_width=0
            ))
            fig.add_trace(go.Scatter(
                x=HOUR_LABELS, y=df["Q_total_required_kw"],
                name="Total Required", mode="lines",
                line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dot")
            ))

            # --- DEBUG FIX: Filter global layout to prevent double-argument error ---
            clean_layout_ccus = {k: v for k, v in PLOT_LAYOUT.items() if k not in ['legend', 'margin']}

            fig.update_layout(
                title="Hourly CCUS Heat Supply (Stacked)",
                barmode="stack", 
                yaxis_title="Heat [kW]", 
                height=340,
                margin=dict(t=100), # Increased top margin for the 4-item legend
                legend=dict(
                    y=1.3,          # Pushes legend above the title
                    x=0.5, 
                    xanchor="center",
                    orientation="h"
                ),
                **clean_layout_ccus
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=HOUR_LABELS, y=df["co2_captured_kgh"],
                fill="tozeroy", fillcolor="rgba(0,212,255,0.1)",
                line=dict(color=COLORS["ccus"], width=2.5),
                name="CO₂ Captured [kg/h]"
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=HOUR_LABELS, y=df["ccus_coverage_pct"],
                line=dict(color=COLORS["credits"], width=2, dash="dash"),
                name="Renewable Coverage [%]"
            ), secondary_y=True)
            fig.update_layout(title="CO₂ Capture Rate & RE Coverage", height=340, **PLOT_LAYOUT)
            fig.update_yaxes(title_text="CO₂ [kg/h]",       secondary_y=False, color=COLORS["ccus"])
            fig.update_yaxes(title_text="RE Coverage [%]",  secondary_y=True,  color=COLORS["credits"])
            st.plotly_chart(fig, use_container_width=True)

        # 20-year projection
        st.markdown('<div class="section-header">20-Year CO₂ & Carbon Credit Projection</div>',
                    unsafe_allow_html=True)
        years       = list(range(1, 21))
        annual_co2  = summary["annual_co2_tonnes"]
        degradation = [annual_co2 * (0.995**y) for y in years]
        cum_co2     = list(np.cumsum(degradation))
        cum_credits = [c * 85 for c in cum_co2]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=years, y=degradation, name="Annual CO₂ [t]",
            marker_color=COLORS["ccus"], opacity=0.75, marker_line_width=0
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=years, y=cum_credits, name="Cumulative Credits [$]",
            line=dict(color=COLORS["credits"], width=2.5),
            fill="tozeroy", fillcolor="rgba(74,222,128,0.06)"
        ), secondary_y=True)
        fig.update_layout(
            title="20-Year CO₂ Capture & Carbon Credit Accumulation",
            xaxis_title="Year", height=320, **PLOT_LAYOUT
        )
        fig.update_yaxes(title_text="CO₂ [tonnes/year]",     secondary_y=False, color=COLORS["ccus"])
        fig.update_yaxes(title_text="Cumulative Credits [$]", secondary_y=True,  color=COLORS["credits"])
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CO₂ Captured/Day",       f"{summary['total_co2_captured_kg']:,.0f} kg")
        c2.metric("Annual CO₂",             f"{summary['annual_co2_tonnes']:,.0f} tonnes")
        c3.metric("Avg CCUS Coverage",      f"{summary['avg_ccus_coverage_pct']:.1f}%")
        c4.metric("20-yr Cumulative CO₂",   f"{sum(degradation):,.0f} t")

    # ────────────────────────────────────────────────────────────
    # TAB 4: BUSINESS CASE
    # ────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Economic Analysis</div>', unsafe_allow_html=True)

        capex_thermal = thermal_area * 250
        capex_pv      = pv_capacity  * 600
        capex_battery = battery_capacity * 300
        total_capex   = capex_thermal + capex_pv + capex_battery

        annual_credits  = summary["annual_credits_usd"]
        annual_grid     = summary["annual_grid_cost_usd"]
        net_annual      = annual_credits - annual_grid
        payback_years   = total_capex / annual_credits if annual_credits > 0 else 99

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total CAPEX",          f"${total_capex:,.0f}",    delta="USD")
        c2.metric("Annual Carbon Credits",f"${annual_credits:,.0f}", delta="USD/year")
        c3.metric("Simple Payback",       f"{payback_years:.1f} yrs")
        c4.metric("Net Annual Benefit",   f"${net_annual:,.0f}",     delta="USD/year")

        col1, col2 = st.columns(2)

        with col1:
            # Payback / cumulative benefit chart
            years         = list(range(0, 21))
            cum_benefits  = [0] + [annual_credits * y for y in range(1, 21)]
            net_cum       = [b - total_capex for b in cum_benefits]
            colors_area   = ["rgba(248,113,113,0.15)" if v < 0 else "rgba(74,222,128,0.15)"
                             for v in net_cum]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years, y=net_cum,
                fill="tozeroy", fillcolor="rgba(74,222,128,0.08)",
                line=dict(color=COLORS["credits"], width=2.5),
                name="Cumulative Net Benefit",
                hovertemplate="Year %{x}<br>Net: $%{y:,.0f}<extra></extra>"
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                          annotation_text=f"Break-even ~{payback_years:.1f} yrs",
                          annotation_font=dict(color="#e2f4ff", size=10))
            fig.update_layout(
                title="20-Year Cumulative Net Benefit",
                xaxis_title="Year", yaxis_title="USD",
                height=320, **PLOT_LAYOUT
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # CAPEX pie
            fig = go.Figure(go.Pie(
                labels=["Solar Thermal", "Solar PV", "Battery Storage"],
                values=[capex_thermal, capex_pv, capex_battery],
                marker=dict(colors=[COLORS["thermal"], COLORS["pv"], COLORS["battery"]],
                            line=dict(color="#020810", width=2)),
                hole=0.55,
                textfont=dict(family="IBM Plex Mono", size=11),
                textinfo="label+percent",
                insidetextorientation="radial"
            ))
            fig.add_annotation(
                text=f"${total_capex/1000:.0f}k<br>CAPEX",
                x=0.5, y=0.5, showarrow=False,
                font=dict(family="IBM Plex Mono", color="#e2f4ff", size=13)
            )
            fig.update_layout(title="Capital Cost Breakdown", height=320, **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # Heat source mix
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Pie(
                labels=["Waste Heat (Free)", "Solar Thermal", "Grid Backup"],
                values=[
                    max(summary["total_waste_heat_kwh"], 0.1),
                    max(summary["total_solar_thermal_kwh"], 0.1),
                    max(summary.get("total_grid_thermal_kwh", 0), 0.1)
                ],
                marker=dict(colors=[COLORS["waste"], COLORS["solar"], COLORS["grid"]],
                            line=dict(color="#020810", width=2)),
                hole=0.55,
                textfont=dict(family="IBM Plex Mono", size=11),
                textinfo="label+percent"
            ))
            fig.add_annotation(
                text="CCUS<br>Heat Mix",
                x=0.5, y=0.5, showarrow=False,
                font=dict(family="IBM Plex Mono", color="#e2f4ff", size=12)
            )
            fig.update_layout(title="CCUS Daily Heat Source Mix", height=300, **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Annual cash flow bar
            years_cf = list(range(1, 21))
            annual_deg = [annual_credits * (0.995**y) for y in years_cf]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=years_cf, y=annual_deg, name="Annual Credits",
                marker_color=[COLORS["credits"] if v > annual_grid else COLORS["grid"]
                             for v in annual_deg],
                marker_line_width=0, opacity=0.8
            ))
            fig.add_hline(y=annual_grid, line_dash="dash", line_color=COLORS["grid"],
                          annotation_text=f"Grid Cost ${annual_grid:,.0f}/yr",
                          annotation_font=dict(color=COLORS["grid"], size=10))
            fig.update_layout(
                title="Annual Carbon Credits vs Grid Cost (20-yr)",
                xaxis_title="Year", yaxis_title="USD",
                height=300, **PLOT_LAYOUT
            )
            st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────────────────────
    # TAB 5: LOCATION COMPARE
    # ────────────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">Global Location Comparison</div>',
                    unsafe_allow_html=True)

        st.info("📍 This tab compares key solar metrics across preset locations using **approximate GHI values** from climatological data. The main simulation uses real NASA POWER API data for your selected location.")

        comparison_data = {
            "Location"               : ["Jodhpur 🇮🇳", "New Delhi 🇮🇳", "Tokyo 🇯🇵", "Osaka 🇯🇵", "Munich 🇩🇪", "Riyadh 🇸🇦", "Phoenix 🇺🇸", "Singapore 🇸🇬"],
            "Annual GHI (kWh/m²/yr)" : [1941,           1627,             1150,          1200,          1050,           2200,            1900,           1600           ],
            "Peak June GHI (W/m²)"   : [826,             720,              550,           580,           680,            900,             820,            600            ],
            "Grid EF (kgCO₂/kWh)"   : [0.82,            0.82,             0.47,          0.47,          0.38,           0.75,            0.45,           0.43           ],
            "Solar Advantage vs Tokyo": ["+69%",         "+42%",           "—",           "+4%",         "-9%",          "+91%",          "+65%",         "+39%"         ],
            "Carbon Price Context"   : ["India Carbon Market","India Carbon Market","J-Credit","J-Credit","EU ETS ~€60","KSA VCM","US VCM","Singapore Carbon Tax"],
        }

        df_comp = pd.DataFrame(comparison_data)
        st.markdown(
            df_comp.to_html(index=False, classes="compare-table"),
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-header">Annual GHI Comparison</div>',
                    unsafe_allow_html=True)

        fig = go.Figure(go.Bar(
            x=comparison_data["Location"],
            y=comparison_data["Annual GHI (kWh/m²/yr)"],
            marker=dict(
                color=comparison_data["Annual GHI (kWh/m²/yr)"],
                colorscale=[[0,"#0369a1"],[0.5,"#0284c7"],[1.0,"#ffb700"]],
                line=dict(color="#020810", width=1)
            ),
            text=comparison_data["Annual GHI (kWh/m²/yr)"],
            texttemplate="%{text} kWh/m²",
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", color="#e2f4ff", size=10),
        ))
        fig.add_hline(y=1150, line_dash="dash", line_color="rgba(255,183,0,0.5)",
                      annotation_text="Tokyo baseline",
                      annotation_font=dict(color="#ffb700", size=10))
        fig.update_layout(
            title="Annual Solar Resource by Location — NASA Climatological Data",
            yaxis_title="GHI [kWh/m²/year]",
            height=360, **PLOT_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
                    border-radius:10px;padding:16px 20px;margin-top:12px;
                    font-family:IBM Plex Sans,sans-serif;font-size:0.82rem;
                    color:#5a8aaa;line-height:1.7;'>
            <strong style='color:#00d4ff;'>Why Jodhpur as the default?</strong><br>
            Jodhpur sits in the Thar Desert — India's premier solar zone with 300+ sunny days/year.
            For a Japanese company building renewable-powered data centers,
            Jodhpur demonstrates <em>maximum system performance</em>.
            Switch to Tokyo or Osaka in the location selector to see how the
            system adapts to Japan's lower irradiance — the optimizer automatically
            recalculates the larger solar thermal area needed to hit the same CCUS coverage target.
        </div>
        """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────
    # TAB 6: FULL OVERVIEW
    # ────────────────────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="section-header">Full System Overview — 6-Panel Dashboard</div>',
                    unsafe_allow_html=True)

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "CPU Load Profile", "DC Waste Heat Generation",
                "Solar PV + Thermal", "Battery SOC",
                "CCUS Heat Stack",   "Combined RE Fraction"
            ],
            vertical_spacing=0.10, horizontal_spacing=0.08
        )

        # R1C1: CPU
        fig.add_trace(go.Scatter(
            x=HOUR_LABELS, y=df["cpu_pct"],
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
            line=dict(color=COLORS["ccus"], width=2), name="CPU %",
            showlegend=False
        ), row=1, col=1)

        # R1C2: Waste heat
        fig.add_trace(go.Bar(
            x=HOUR_LABELS, y=df["dc_waste_heat_kw"],
            marker_color=COLORS["thermal"], name="Waste Heat",
            marker_line_width=0, showlegend=False
        ), row=1, col=2)

        # R2C1: Solar
        fig.add_trace(go.Bar(
            x=HOUR_LABELS, y=df["P_pv_kw"],
            marker_color=COLORS["pv"], name="PV",
            marker_line_width=0, showlegend=False
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=HOUR_LABELS, y=df["Q_thermal_kw"],
            marker_color=COLORS["thermal"], name="Thermal",
            marker_line_width=0, showlegend=False
        ), row=2, col=1)

        # R2C2: Battery
        fig.add_trace(go.Scatter(
            x=HOUR_LABELS, y=df["soc_pct"],
            fill="tozeroy", fillcolor="rgba(168,85,247,0.1)",
            line=dict(color=COLORS["battery"], width=2), name="SOC",
            showlegend=False
        ), row=2, col=2)

        # R3C1: CCUS stack
        for name, col, key in [
            ("Waste", COLORS["waste"], "Q_waste_heat_kw"),
            ("Solar", COLORS["solar"], "Q_solar_to_ccus_kw"),
            ("Grid",  COLORS["grid"],  "Q_ccus_deficit_kw"),
        ]:
            fig.add_trace(go.Bar(
                x=HOUR_LABELS, y=df[key],
                marker_color=col, name=name,
                marker_line_width=0, showlegend=False
            ), row=3, col=1)

        # R3C2: RE fraction
        fig.add_trace(go.Scatter(
            x=HOUR_LABELS, y=df["combined_re_pct"],
            fill="tozeroy", fillcolor="rgba(74,222,128,0.08)",
            line=dict(color=COLORS["credits"], width=2), name="RE %",
            showlegend=False
        ), row=3, col=2)

        fig.update_layout(
            height=760, showlegend=False, barmode="stack",
            **PLOT_LAYOUT
        )
        for ann in fig.layout.annotations:
            ann.font.update(family="IBM Plex Mono", size=11, color="#5a8aaa")

        st.plotly_chart(fig, use_container_width=True)

        # Daily summary table
        st.markdown('<div class="section-header">Daily Summary Statistics</div>',
                    unsafe_allow_html=True)
        summary_display = {
            "Metric": [
                "Average CPU Load", "Total Facility Energy", "Total Waste Heat Generated",
                "Waste Heat Captured", "Solar PV Generation", "Solar Thermal Generation",
                "Grid Import (Electrical)", "Grid Thermal Backup",
                "CO₂ Captured", "Carbon Credits Earned", "Net Benefit"
            ],
            "Value": [
                f"{summary['avg_cpu_pct']:.1f}%",
                f"{summary['total_grid_import_kwh'] + summary['total_pv_kwh']:.0f} kWh",
                f"{sum(df['dc_waste_heat_kw']):.0f} kWh",
                f"{summary['total_waste_heat_kwh']:.0f} kWh",
                f"{summary['total_pv_kwh']:.0f} kWh",
                f"{summary['total_solar_thermal_kwh']:.0f} kWh",
                f"{summary['total_grid_import_kwh']:.0f} kWh",
                f"{summary.get('total_grid_thermal_kwh', 0):.0f} kWh",
                f"{summary['total_co2_captured_kg']:,.0f} kg",
                f"${summary['total_carbon_credits_usd']:.2f}",
                f"${summary['net_benefit_usd_day']:.2f}"
            ],
            "Annual": [
                "—",
                f"{(summary['total_grid_import_kwh'] + summary['total_pv_kwh'])*365:,.0f} kWh",
                "—", "—",
                f"{summary['total_pv_kwh']*365:,.0f} kWh",
                f"{summary['total_solar_thermal_kwh']*365:,.0f} kWh",
                f"{summary['total_grid_import_kwh']*365:,.0f} kWh",
                "—",
                f"{summary['annual_co2_tonnes']:,.1f} tonnes",
                f"${summary['annual_credits_usd']:,.0f}",
                f"${summary['net_benefit_usd_day']*365:,.0f}"
            ]
        }
        df_summary = pd.DataFrame(summary_display)
        st.markdown(
            df_summary.to_html(index=False, classes="compare-table"),
            unsafe_allow_html=True
        )

# ============================================================================
# FOOTER
# ============================================================================

def render_footer():
    st.markdown("""
    <div class="footer">
        GX×DX CLOSED-LOOP DIGITAL TWIN &nbsp;·&nbsp;
        V NIRANJANA · B.TECH CHEMICAL ENGINEERING · IIT JODHPUR &nbsp;·&nbsp;
        MODULES: DATACENTER THERMAL · HEAT EXCHANGER (LMTD) · MEA-CCUS · SOLAR+BATTERY · OPTIMIZER &nbsp;·&nbsp;
        DATA: NASA POWER API
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN
# ============================================================================

def main():
    (pue, cpu_profile, thermal_area, pv_capacity,
     battery_capacity, month, mea_flow,
     lat, lon, location_key, grid_ef) = render_sidebar()

    render_hero(location_key, lat, lon, month)

    with st.spinner("⚡ Running full system simulation..."):
        result, ghi_profile, cpu_p = run_simulation(
            cpu_profile, thermal_area, pv_capacity,
            battery_capacity, month, pue, mea_flow,
            lat, lon, location_key
        )

    hourly  = result["hourly"]
    summary = result["summary"]

    render_kpis(summary, thermal_area, pv_capacity, battery_capacity)

    col1, col2 = st.columns([3, 2])
    with col1:
        render_pipeline(hourly, summary)
    with col2:
        st.markdown(
            f'<div style="padding-top:28px;">'
            f'<span class="status-badge"><span class="status-dot"></span>'
            f'SIMULATION LIVE — {MONTH_NAMES[month-1].upper()} PROFILE</span>'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
            f'color:#2a4a62;margin-top:10px;line-height:1.8;">'
            f'CPU PROFILE: {cpu_profile.upper()}<br>'
            f'PUE: {pue} · MEA: {mea_flow} kg/s<br>'
            f'SOLAR THERMAL: {thermal_area} m²<br>'
            f'PV: {pv_capacity} kWp · BATT: {battery_capacity} kWh'
            f'</div></div>',
            unsafe_allow_html=True
        )

    render_sankey(hourly)
    render_tabs(hourly, summary, ghi_profile, cpu_p, location_key, thermal_area, pv_capacity, battery_capacity)
    render_footer()

if __name__ == "__main__":
    main()