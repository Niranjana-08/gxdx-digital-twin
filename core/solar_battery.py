"""
=============================================================================
MODULE 4: Solar PV + Battery Storage Model
=============================================================================
Project : GX×DX Closed-Loop Digital Twin
Author  : V Niranjana | IIT Jodhpur
Purpose : Models solar PV electricity generation and battery storage to
          power both the data center and supplement the CCUS solar thermal
          heat requirement.

          Uses NASA POWER API for real hourly solar irradiance data.
          Falls back to synthetic clear-sky model if API unavailable.

Two energy streams modelled:
  1. Solar PV → Electricity → Powers data center (offsets grid import)
  2. Solar Thermal → Heat → Supplements MEA regenerator deficit

Key equations:
  P_pv(t)   = G(t) × A × η_panel × η_inv          [kW electrical]
  Q_st(t)   = G(t) × A_thermal × η_thermal         [kW thermal]
  SOC(t)    = SOC(t-1) ± P_net × Δt / E_capacity   [kWh/kWh]
=============================================================================
"""

import numpy as np
import requests
import json
import os

# ---------------------------------------------------------------------------
# SYSTEM CONSTANTS
# ---------------------------------------------------------------------------

# Solar PV panel parameters
PANEL_EFFICIENCY    = 0.20      # 20% — modern monocrystalline silicon
INVERTER_EFFICIENCY = 0.97      # 97% — standard string inverter
PANEL_TEMP_COEFF    = -0.004    # -0.4%/°C above 25°C (STC)
PANEL_AREA_M2       = 2.0       # m² per panel (standard 400W panel)

# Solar thermal collector parameters
THERMAL_EFFICIENCY  = 0.55      # 55% — flat-plate collector (conservative)
                                 # Evacuated tube: up to 70%

# Battery parameters
BATTERY_EFFICIENCY  = 0.92      # Round-trip efficiency (Li-ion)
SOC_MIN             = 0.10      # 10% minimum state of charge (protect battery)
SOC_MAX             = 0.95      # 95% maximum (prevent overcharge)
SELF_DISCHARGE      = 0.0002    # 0.02% per hour self-discharge

# Economics
GRID_ELECTRICITY_COST = 0.08    # USD/kWh — industrial tariff India
SOLAR_CAPEX_PER_KWP   = 600     # USD/kWp — utility solar India 2024

# ---------------------------------------------------------------------------
# NASA POWER API
# ---------------------------------------------------------------------------

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

def fetch_nasa_solar(lat: float, lon: float, year: int = 2023) -> list:
    """
    Fetch hourly solar irradiance (GHI) from NASA POWER API.
    Returns list of 8760 hourly GHI values [W/m²] for the full year.

    Falls back to synthetic model if API call fails.

    Parameters
    ----------
    lat : float   Latitude  (e.g. 26.9 for Jodhpur, India)
    lon : float   Longitude (e.g. 73.0 for Jodhpur, India)
    year : int    Year for historical data. Default 2023.
    """
    print(f"  Fetching NASA POWER solar data for ({lat}, {lon}), year {year}...")

    params = {
        "parameters" : "ALLSKY_SFC_SW_DWN",   # Global Horizontal Irradiance
        "community"  : "RE",                   # Renewable Energy community
        "longitude"  : lon,
        "latitude"   : lat,
        "start"      : f"{year}0101",
        "end"        : f"{year}1231",
        "format"     : "JSON",
        "header"     : "true",
        "time-standard": "LST",               # Local Solar Time
    }

    try:
        response = requests.get(NASA_POWER_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract GHI values — returns dict keyed by "YYYYMMDDHH"
        ghi_dict = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

        # Convert to ordered list of 8760 hourly values [W/m²]
        ghi_values = [max(0, v) for v in ghi_dict.values()]

        # NASA uses -999 for missing data — replace with 0
        ghi_values = [0 if v < 0 else v for v in ghi_values]

        print(f"  ✅ NASA data fetched: {len(ghi_values)} hourly values")
        print(f"     Annual GHI total: {sum(ghi_values)/1000:.0f} kWh/m²/year")

        # Save to cache
        cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "nasa_solar_data.json"
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"lat": lat, "lon": lon, "year": year,
                       "ghi_wm2": ghi_values}, f)

        return ghi_values

    except Exception as e:
        print(f"  ⚠️  NASA API unavailable ({e}). Using synthetic clear-sky model.")
        return _synthetic_solar_profile(lat)


def _synthetic_solar_profile(lat: float = 26.9) -> list:
    """
    Generate synthetic 8760-hour solar GHI profile using clear-sky model.
    Based on sinusoidal approximation of daily and seasonal variation.
    Used as fallback when NASA API is unavailable.
    """
    ghi = []
    for day in range(365):
        # Seasonal factor: peak in summer (day 172 = June 21)
        seasonal = 0.75 + 0.25 * np.cos(2 * np.pi * (day - 172) / 365)

        # Latitude correction: higher lat = lower irradiance
        lat_factor = 1.0 - abs(lat - 20) * 0.005

        peak_ghi = 900 * seasonal * lat_factor  # W/m² peak

        for hour in range(24):
            # Solar hours: 6am to 6pm (hours 6–18)
            if 6 <= hour <= 18:
                # Bell curve centred at solar noon (hour 12)
                solar_angle = np.pi * (hour - 6) / 12
                ghi_val     = peak_ghi * np.sin(solar_angle) ** 1.2
            else:
                ghi_val = 0.0
            ghi.append(max(0.0, ghi_val))

    return ghi


def get_typical_day(ghi_annual: list, month: int = 6) -> list:
    """
    Extract a representative 24-hour GHI profile for a given month.
    Returns list of 24 hourly GHI values [W/m²].

    Parameters
    ----------
    ghi_annual : list   8760-hour annual GHI values
    month      : int    Month (1=Jan, 6=Jun). Default = June (peak solar).
    """
    # Month start days (non-leap year)
    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    start_day = month_starts[month - 1]
    n_days    = month_lengths[month - 1]

    # Average GHI across all days in the month, hour by hour
    hourly_avg = []
    for hour in range(24):
        hour_vals = []
        for day in range(start_day, start_day + n_days):
            idx = day * 24 + hour
            if idx < len(ghi_annual):
                hour_vals.append(ghi_annual[idx])
        hourly_avg.append(np.mean(hour_vals) if hour_vals else 0.0)

    return hourly_avg


# ---------------------------------------------------------------------------
# CORE CLASS
# ---------------------------------------------------------------------------

class SolarBatterySystem:
    """
    Models a solar PV array + battery storage system that:
      1. Generates electricity to power the data center
      2. Generates solar thermal heat to supplement CCUS regenerator

    Parameters
    ----------
    pv_capacity_kwp : float
        Installed solar PV capacity [kWp]. Default = 150 kWp.
        (Sized to approximately cover 100kW DC at peak sun hours)
    thermal_area_m2 : float
        Solar thermal collector area [m²]. Default = 300 m².
        (Sized to approximately cover MEA heat deficit at peak)
    battery_capacity_kwh : float
        Battery storage capacity [kWh]. Default = 200 kWh.
        (Approximately 2 hours of DC load)
    battery_power_kw : float
        Max charge/discharge power [kW]. Default = 100 kW (1C rate).
    location : str
        Human-readable location name for display.
    """

    def __init__(
        self,
        pv_capacity_kwp      = 150.0,
        thermal_area_m2      = 300.0,
        battery_capacity_kwh = 200.0,
        battery_power_kw     = 100.0,
        location             = "Jodhpur, India"
    ):
        self.pv_capacity_kwp      = pv_capacity_kwp
        self.thermal_area_m2      = thermal_area_m2
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_power_kw     = battery_power_kw
        self.location             = location

        # Derive PV array area from capacity
        # 1 kWp requires ~5 m² of panel at 20% efficiency, 1000 W/m² STC
        self.pv_area_m2 = pv_capacity_kwp / (PANEL_EFFICIENCY * 1.0)  # m²

    # ------------------------------------------------------------------
    def compute_hour(
        self,
        ghi_wm2:        float,
        dc_demand_kw:   float,
        soc_prev:       float,
        ambient_temp_c: float = 28.0
    ) -> dict:
        """
        Compute solar generation, battery state, and energy balance
        for a single hour.

        Parameters
        ----------
        ghi_wm2 : float
            Global horizontal irradiance [W/m²] for this hour.
        dc_demand_kw : float
            Data center electrical demand for this hour [kW].
        soc_prev : float
            Battery state of charge at start of hour [0–1].
        ambient_temp_c : float
            Ambient air temperature [°C].

        Returns
        -------
        dict with generation, storage, and balance results.
        """

        ghi_kw_m2 = ghi_wm2 / 1000.0   # Convert W/m² → kW/m²

        # --- Solar PV generation ---
        # Temperature derating: panels lose efficiency when hot
        # T_cell ≈ T_ambient + 25°C (typical NOCT correction)
        T_cell         = ambient_temp_c + 25.0
        temp_derate    = 1 + PANEL_TEMP_COEFF * (T_cell - 25.0)
        temp_derate    = max(temp_derate, 0.5)   # floor at 50% (safety)

        P_pv_kw = (ghi_kw_m2 * self.pv_area_m2 *
                   PANEL_EFFICIENCY * INVERTER_EFFICIENCY * temp_derate)
        P_pv_kw = max(P_pv_kw, 0.0)

        # --- Solar thermal generation ---
        # Q_thermal = G × A × η_thermal
        Q_thermal_kw = ghi_kw_m2 * self.thermal_area_m2 * THERMAL_EFFICIENCY
        Q_thermal_kw = max(Q_thermal_kw, 0.0)

        # --- Net electrical balance ---
        # Positive = surplus (charge battery / export)
        # Negative = deficit (discharge battery / import grid)
        P_net_kw = P_pv_kw - dc_demand_kw

        # --- Battery operation ---
        soc = soc_prev * (1 - SELF_DISCHARGE)   # self-discharge first

        if P_net_kw >= 0:
            # Surplus → charge battery
            charge_kw    = min(P_net_kw, self.battery_power_kw)
            charge_kwh   = charge_kw * BATTERY_EFFICIENCY   # efficiency loss
            soc_new      = soc + charge_kwh / self.battery_capacity_kwh
            soc_new      = min(soc_new, SOC_MAX)
            actual_charge_kw = charge_kw
            actual_discharge_kw = 0.0
            # Power actually surplus after charging
            P_export_kw  = P_net_kw - charge_kw
            P_grid_kw    = 0.0   # no grid needed
        else:
            # Deficit → discharge battery
            deficit_kw   = abs(P_net_kw)
            discharge_kw = min(deficit_kw, self.battery_power_kw)
            discharge_kwh= discharge_kw / BATTERY_EFFICIENCY
            soc_new      = soc - discharge_kwh / self.battery_capacity_kwh
            if soc_new < SOC_MIN:
                # Battery depleted — need grid import for remainder
                available_kwh = max(0, (soc - SOC_MIN) * self.battery_capacity_kwh)
                actual_discharge_kw = available_kwh * BATTERY_EFFICIENCY
                soc_new = SOC_MIN
            else:
                actual_discharge_kw = discharge_kw
            actual_charge_kw = 0.0
            P_export_kw  = 0.0
            # Grid makes up what battery couldn't cover
            P_grid_kw    = max(0, deficit_kw - actual_discharge_kw)

        # --- Renewable fraction for this hour ---
        renewable_elec = P_pv_kw + actual_discharge_kw
        total_demand   = dc_demand_kw if dc_demand_kw > 0 else 0.001
        renewable_frac = min(renewable_elec / total_demand, 1.0)

        # --- CO₂ saved by renewables (vs grid) ---
        INDIA_GRID_EF   = 0.82   # kgCO₂/kWh
        co2_saved_kg    = (P_pv_kw + actual_discharge_kw - P_grid_kw) * INDIA_GRID_EF
        co2_saved_kg    = max(co2_saved_kg, 0.0)

        # --- Cost savings ---
        grid_cost_saved = (P_pv_kw + actual_discharge_kw) * GRID_ELECTRICITY_COST

        return {
            # Generation
            "P_pv_kw"              : round(P_pv_kw, 2),
            "Q_thermal_kw"         : round(Q_thermal_kw, 2),
            "ghi_wm2"              : round(ghi_wm2, 1),
            "temp_derate_pct"      : round(temp_derate * 100, 1),

            # Battery
            "soc_start"            : round(soc_prev, 3),
            "soc_end"              : round(soc_new, 3),
            "charge_kw"            : round(actual_charge_kw, 2),
            "discharge_kw"         : round(actual_discharge_kw, 2),

            # Energy balance
            "dc_demand_kw"         : round(dc_demand_kw, 2),
            "P_grid_import_kw"     : round(P_grid_kw, 2),
            "P_export_kw"          : round(P_export_kw, 2),
            "renewable_frac_pct"   : round(renewable_frac * 100, 1),

            # CCUS support
            "Q_thermal_for_ccus_kw": round(Q_thermal_kw, 2),  # available for MEA

            # Environment & economics
            "co2_saved_kg"         : round(co2_saved_kg, 2),
            "grid_cost_saved_usd"  : round(grid_cost_saved, 4),
        }

    # ------------------------------------------------------------------
    def simulate_day(
        self,
        ghi_profile:    list,
        dc_daily:       list,
        ambient_temps:  list = None,
        soc_initial:    float = 0.50
    ) -> list:
        """
        Simulate 24-hour operation.

        Parameters
        ----------
        ghi_profile   : list of 24 GHI values [W/m²]
        dc_daily      : list of 24 DC result dicts (from DataCenterThermalModel)
        ambient_temps : list of 24 ambient temperatures [°C]. 
                        Defaults to typical Jodhpur June profile.
        soc_initial   : float, starting battery SOC [0–1]. Default 0.50.
        """
        if ambient_temps is None:
            # Typical Jodhpur June temperature profile [°C]
            ambient_temps = [
                28, 27, 26, 26, 27, 29, 32, 35, 38, 40,
                41, 42, 42, 41, 40, 39, 38, 36, 34, 32,
                31, 30, 29, 28
            ]

        results = []
        soc     = soc_initial

        for hour in range(24):
            r = self.compute_hour(
                ghi_wm2        = ghi_profile[hour],
                dc_demand_kw   = dc_daily[hour]["total_facility_power_kw"],
                soc_prev       = soc,
                ambient_temp_c = ambient_temps[hour]
            )
            soc = r["soc_end"]
            results.append(r)

        return results


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from core.datacenter_thermal import DataCenterThermalModel, get_cpu_profile

    print("=" * 65)
    print("  SOLAR + BATTERY MODEL — QUICK TEST")
    print("=" * 65)
    print(f"\n  Location: Jodhpur, India (Lat: 26.9°N, Lon: 73.0°E)")
    print(f"  PV Capacity    : 150 kWp")
    print(f"  Thermal Area   : 300 m²")
    print(f"  Battery        : 200 kWh / 100 kW")

    # --- Fetch NASA solar data ---
    print()
    ghi_annual = fetch_nasa_solar(lat=26.9, lon=73.0, year=2023)

    # --- Get typical June day (peak solar month for Jodhpur) ---
    ghi_june   = get_typical_day(ghi_annual, month=6)

    print(f"\n  Peak GHI (June typical day): {max(ghi_june):.0f} W/m²")
    print(f"  Daily solar energy (June)  : {sum(ghi_june)/1000:.2f} kWh/m²/day")

    # --- DC simulation ---
    dc      = DataCenterThermalModel()
    profile = get_cpu_profile("office")
    daily_dc= dc.simulate_day(profile)

    # --- Solar + Battery simulation ---
    solar   = SolarBatterySystem(
        pv_capacity_kwp      = 150.0,
        thermal_area_m2      = 300.0,
        battery_capacity_kwh = 200.0,
        battery_power_kw     = 100.0
    )

    daily_solar = solar.simulate_day(ghi_june, daily_dc)

    # --- Print hourly results ---
    print(f"\n📅 24-Hour Solar + Battery Performance (June Typical Day)")
    print("-" * 85)
    print(f"  {'Hr':>3} {'GHI':>6} {'P_PV':>7} {'Q_therm':>8} {'DC_dem':>7} "
          f"{'Grid':>7} {'SOC%':>6} {'RE%':>6} {'CO2sv':>7}")
    print(f"  {'─'*3} {'─'*6} {'─'*7} {'─'*8} {'─'*7} "
          f"{'─'*7} {'─'*6} {'─'*6} {'─'*7}")

    total_pv      = 0; total_thermal = 0; total_grid  = 0
    total_co2_sv  = 0; total_savings = 0

    for hr, (dc_r, sol_r) in enumerate(zip(daily_dc, daily_solar)):
        total_pv      += sol_r["P_pv_kw"]
        total_thermal += sol_r["Q_thermal_kw"]
        total_grid    += sol_r["P_grid_import_kw"]
        total_co2_sv  += sol_r["co2_saved_kg"]
        total_savings += sol_r["grid_cost_saved_usd"]

        print(
            f"  {hr:02d}:00"
            f"  {sol_r['ghi_wm2']:>5.0f}W"
            f"  {sol_r['P_pv_kw']:>6.1f}kW"
            f"  {sol_r['Q_thermal_kw']:>7.1f}kW"
            f"  {sol_r['dc_demand_kw']:>6.1f}kW"
            f"  {sol_r['P_grid_import_kw']:>6.1f}kW"
            f"  {sol_r['soc_end']*100:>5.1f}%"
            f"  {sol_r['renewable_frac_pct']:>5.1f}%"
            f"  {sol_r['co2_saved_kg']:>6.1f}kg"
        )

    print(f"\n  {'─'*65}")
    print(f"  Total PV generation (24h)      : {total_pv:.1f} kWh")
    print(f"  Total solar thermal (24h)      : {total_thermal:.1f} kWh")
    print(f"  Total grid import (24h)        : {total_grid:.1f} kWh")
    print(f"  Total CO₂ saved (electricity)  : {total_co2_sv:.1f} kg")
    print(f"  Total electricity savings      : ${total_savings:.2f}")
    print(f"  Solar thermal vs CCUS need     : {total_thermal:.1f} / 4755 kWh needed")
    print(f"  Solar thermal coverage of CCUS : {total_thermal/4755*100:.1f}%")
    print(f"\n  ✅ Module 4 complete — ready to build optimizer.py")