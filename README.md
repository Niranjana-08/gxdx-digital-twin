
# ⚡ GX×DX Closed-Loop Digital Twin

This repository contains an end-to-end **physics-based simulation engine** designed to model, optimize, and visualize the integration of a renewable-powered modular data center with a **MEA-based CO₂ capture unit**. The core of the project involves coupling five thermodynamic modules — from server waste heat recovery to solar-driven CCUS regeneration — closing the loop between **Digital Transformation (DX)** and **Green Transformation (GX)**.

The culmination of this work is a professional **SCADA-style Dashboard** built with Streamlit, which allows engineers and researchers to simulate, size, and evaluate the system for any location on Earth using real NASA solar data.

<br>

## 🌍 The Problem This Solves

Data centers waste **massive amounts of heat** — a 100 kW server room rejects over 1,600 kWh of thermal energy into the atmosphere every single day. At the same time, CCUS systems require **massive amounts of heat** to strip CO₂ from MEA solvent at 120°C.

This project **closes that loop.**

Waste heat from servers pre-heats the MEA solvent. Solar thermal collectors bridge the remaining temperature gap. Solar PV powers the data center. A Python optimizer finds the minimum system size to hit any renewable coverage target — all driven by **real NASA solar irradiance data** for any location on Earth.

```
☀️ Solar PV + Battery  →  🖥️ 100kW Data Center  →  ♨️ Heat Exchanger
                                                            ↓
                       ☀️ Solar Thermal Collectors  →  🏭 MEA Regenerator
                                                            ↓
                                         ✅ CO₂ Captured · 💰 Carbon Credits Generated
```

<br>

## <img src="https://user-images.githubusercontent.com/106439762/181935629-b3c47bd3-77fb-4431-a11c-ff8ba0942b63.gif" height=40 width=40> Project Structure

| File / Folder | Description |
| :--- | :--- |
| `core/datacenter_thermal.py` | **The DC Thermal Engine:** Models 100kW data center heat output vs CPU load, PUE, and airflow across 24h workload profiles. |
| `core/heat_exchanger.py` | **The Heat Exchanger:** LMTD + NTU-effectiveness counterflow model for glycol-water loop sizing and performance rating. |
| `core/mea_regenerator.py` | **The CCUS Module:** Thermodynamic model of CO₂ stripping from MEA solvent — reboiler duty, capture rate, and carbon credits. |
| `core/solar_battery.py` | **Solar + Battery:** Fetches real NASA POWER irradiance data, models PV generation with temperature derating and Li-ion battery SOC. |
| `core/optimizer.py` | **The System Optimizer:** Integrates all modules, runs hourly energy balance, and sweeps solar thermal area for target RE coverage. |
| `dashboard/app.py` | **The Dashboard:** Main Streamlit application handling UI, live simulation, visualization, and user interaction. |
| `data/nasa_solar_data.json` | Cached NASA POWER hourly GHI data — auto-generated on first run per location. |
| `outputs/pfd_diagram.svg` | Process Flow Diagram with equipment tags, stream numbers, and instrumentation symbols. |
| `requirements.txt` | A list of Python libraries and dependencies required to run the project. |

<br>

## <img src="https://user-images.githubusercontent.com/106439762/181937125-2a4b22a3-f8a9-4226-bbd3-df972f9dbbc4.gif" height=40 width=40> Tools & Technology Used

<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3776AB.svg?logo=python&logoColor=white"></a>
<a href="#"><img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white"></a>
<a href="#"><img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243.svg?logo=numpy&logoColor=white"></a>
<a href="#"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white"></a>
<a href="#"><img alt="Plotly" src="https://img.shields.io/badge/Plotly-3F4F75.svg?logo=plotly&logoColor=white"></a>
<a href="#"><img alt="NASA API" src="https://img.shields.io/badge/NASA%20POWER%20API-0B3D91.svg?logo=nasa&logoColor=white"></a>

<br>

## <img src="https://user-images.githubusercontent.com/106439762/178428775-03d67679-9aa4-4b08-91e9-6eb6ed8faf66.gif" height=40 width=40> Project Pipeline

#### 1️⃣ Data Center Thermal Engine
<details>
<summary>Model how server workload generates recoverable waste heat.</summary>

* Simulates a **100 kW modular data center** with configurable PUE (Power Usage Effectiveness).
* Generates 24-hour waste heat profiles across three CPU workload patterns — office hours, cloud always-on, and night batch.
* Computes hourly rack outlet temperature and total recoverable thermal energy passed downstream.
</details>

#### 2️⃣ Shell & Tube Heat Exchanger
<details>
<summary>Transfer server waste heat into the CCUS glycol pre-heat loop.</summary>

* Implements the **LMTD method** (Log Mean Temperature Difference) for counterflow heat exchanger sizing.
* Uses **NTU-Effectiveness** method for performance rating given area and flow conditions.
* Delivers pre-heated glycol (25→34°C) from server exhaust (46°C) — bridging the first thermal gap toward MEA regeneration.
</details>

#### 3️⃣ MEA Regenerator — CCUS Thermodynamics
<details>
<summary>Model the energy demand of stripping CO₂ from MEA solvent.</summary>

* Computes sensible heat, reaction enthalpy (85 kJ/mol CO₂), and vaporization losses across the reboiler.
* Validates **specific reboiler duty at 3.59 GJ/tonne CO₂** — within the industry benchmark of 3.5–4.5.
* Models 90% CO₂ capture efficiency on a 1,000 Nm³/h flue gas stream at 12% CO₂ concentration.
</details>

#### 4️⃣ Solar PV + Battery Storage
<details>
<summary>Power the data center and supply thermal energy from renewables.</summary>

* Fetches **real 8,760-hour NASA POWER API irradiance data** for any lat/lon on Earth.
* Models temperature-derated PV generation and Li-ion battery charge/discharge with SOC tracking.
* Simulates flat-plate solar thermal collectors feeding the MEA reboiler heat deficit directly.
</details>

#### 5️⃣ System Optimizer
<details>
<summary>Find the minimum renewable system size to hit any CO₂ coverage target.</summary>

* Integrates all four modules into a single **hourly energy balance** across all 24 hours.
* Sweeps solar thermal area (100–1500 m²) to find the optimal size for a user-defined RE coverage target.
* Outputs carbon credit revenue, grid cost, net annual benefit, and simple payback period.
</details>

<br>

## <img src="https://camo.githubusercontent.com/a2daec8e86875076d58c5445979eaa0994a4edc2e415b70108c429526572dc04/68747470733a2f2f63646e312e766563746f7273746f636b2e636f6d2f692f3130303078313030302f34352f37302f64617368626f6172642d69636f6e2d766563746f722d32323839343537302e6a7067" height=40 width=40> Interactive Dashboard

An interactive Streamlit dashboard brings the full simulation to life — no code required. Adjust any system parameter in real time and watch every chart, flow diagram, and economic output update instantly.

### 🌍 Location-Aware Simulation
Switch between **8 global city presets** (Tokyo, Jodhpur, Munich, Riyadh, Singapore and more) or enter any custom lat/lon. The system fetches real NASA solar data and recalibrates the entire model for that location automatically.

### The Main Dashboard
Features a dark SCADA-style interface with a live Sankey energy flow diagram, 6 KPI cards, and a real-time system pipeline showing energy flows at peak hour.

### 6-Tab Interface

| Tab | What It Shows |
| :--- | :--- |
| 🖥️ **Data Center** | Hourly power & waste heat vs CPU load · Glycol loop temperature · Heat recovery metrics |
| ☀️ **Solar & Battery** | PV + thermal generation vs NASA irradiance · Battery SOC · Grid import · RE fraction |
| 🏭 **CCUS** | Stacked heat supply (waste + solar + grid backup) · CO₂ capture rate · 20-year projection |
| 💰 **Business Case** | CAPEX breakdown · Payback curve · Carbon credit vs grid cost · Heat source mix |
| 🌍 **Location Compare** | GHI comparison across 8 global cities · Solar resource advantage quantified |
| 📊 **Full Overview** | 6-panel system dashboard · Complete daily summary statistics table |

<br>

## <img src="https://user-images.githubusercontent.com/106439762/178803205-47a08ce7-2187-4f96-b301-a2b68690619a.gif" height=40 width=40> Quick Start

**Try the live dashboard — no installation needed:**

> 🚀 **[Launch Interactive Dashboard →](https://gxdx-digital-twin.streamlit.app/)**

To run locally:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/gxdx-digital-twin.git
cd gxdx-digital-twin

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run dashboard/app.py
```

To test individual physics modules:
```bash
python core/optimizer.py    # runs full pipeline and prints daily summary
```

<br>

## <img src="https://user-images.githubusercontent.com/108053296/185756908-fbb62168-d923-48f2-992f-b8e2fde848fe.gif" height=40 width=40> Engineering Impact

* **Real Data, Not Assumptions:** Every simulation runs on live NASA POWER irradiance data — switch cities and the whole system recalibrates with historically accurate solar profiles.
* **Globally Deployable:** The same physics engine works for Jodhpur, Tokyo, Munich, or any custom coordinates — quantifying exactly how location affects system sizing and economics.
* **End-to-End Integration:** Five physics modules communicate through a single optimizer, making this a true digital twin rather than an isolated calculator.
* **Industry Validated:** MEA specific reboiler duty of 3.59 GJ/tonne CO₂ sits within the industry benchmark of 3.5–4.5 — the thermodynamic model is physically grounded, not curve-fitted.

<br>

## Check out the live interactive dashboard here: [GX×DX Digital Twin](https://gxdx-digital-twin.streamlit.app/)

<br>

---
