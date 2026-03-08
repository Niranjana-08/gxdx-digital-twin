"""
=============================================================================
MODULE 5: System Optimizer — GX×DX Closed-Loop Energy Balancer
=============================================================================
Project : GX×DX Closed-Loop Digital Twin
Author  : V Niranjana | IIT Jodhpur
Purpose : Integrates all four modules into a complete system simulation.
          Runs hourly energy balance across the full pipeline:

            Solar PV + Battery → Data Center → Heat Exchanger → MEA CCUS

          Two optimization functions:
          1. simulate_system()   — full 24h system simulation at fixed sizing
          2. optimize_sizing()   — finds minimum solar thermal area to hit
                                   a target renewable coverage of CCUS heat

Key Output Metrics:
  - Hourly energy flows across all components
  - Renewable energy fraction (electricity + thermal)
  - CO₂ captured and carbon credits generated
  - Grid dependency and energy cost
  - System LCOE and payback period
=============================================================================
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.datacenter_thermal import DataCenterThermalModel, get_cpu_profile
from core.heat_exchanger      import HeatExchanger
from core.mea_regenerator     import MEARegenerator
from core.solar_battery       import SolarBatterySystem, fetch_nasa_solar, get_typical_day

# ---------------------------------------------------------------------------
# ECONOMICS CONSTANTS
# ---------------------------------------------------------------------------

GRID_ELECTRICITY_USD_KWH  = 0.08     # USD/kWh industrial India
SOLAR_THERMAL_CAPEX_M2    = 250.0    # USD/m² installed solar thermal
SOLAR_PV_CAPEX_KWP        = 600.0    # USD/kWp installed solar PV
BATTERY_CAPEX_KWH         = 300.0    # USD/kWh battery storage
CARBON_CREDIT_USD_TONNE   = 85.0     # USD/tonne CO₂
SYSTEM_LIFETIME_YEARS     = 20       # years
DISCOUNT_RATE             = 0.08     # 8% — typical for India renewable projects
SOLAR_DEGRADATION         = 0.005    # 0.5%/year panel degradation

# ---------------------------------------------------------------------------
# CORE CLASS
# ---------------------------------------------------------------------------

class SystemOptimizer:
    """
    Full GX×DX system integrator and optimizer.

    Parameters
    ----------
    dc_model    : DataCenterThermalModel instance
    hx_model    : HeatExchanger instance
    mea_model   : MEARegenerator instance
    solar_model : SolarBatterySystem instance
    """

    def __init__(self, dc_model, hx_model, mea_model, solar_model):
        self.dc    = dc_model
        self.hx    = hx_model
        self.mea   = mea_model
        self.solar = solar_model

    # ------------------------------------------------------------------
    def simulate_system(
        self,
        cpu_profile:   list,
        ghi_profile:   list,
        ambient_temps: list = None,
        soc_initial:   float = 0.50
    ) -> dict:
        """
        Run full 24-hour system simulation.

        Parameters
        ----------
        cpu_profile   : list of 24 CPU utilization values [0–1]
        ghi_profile   : list of 24 GHI values [W/m²]
        ambient_temps : list of 24 ambient temperatures [°C]
        soc_initial   : initial battery SOC [0–1]

        Returns
        -------
        dict with:
          'hourly'  : list of 24 combined result dicts
          'summary' : aggregated daily KPIs
        """

        if ambient_temps is None:
            ambient_temps = [
                28, 27, 26, 26, 27, 29, 32, 35, 38, 40,
                41, 42, 42, 41, 40, 39, 38, 36, 34, 32,
                31, 30, 29, 28
            ]

        hourly_results = []
        soc = soc_initial

        for hour in range(24):

            # --- Layer 1: Data Center ---
            dc_r = self.dc.compute(cpu_profile[hour])

            # --- Layer 2: Heat Exchanger ---
            hx_r = self.hx.rate(dc_r)

            # --- Layer 3: MEA Regenerator ---
            mea_r = self.mea.compute(hx_r)

            # --- Layer 4: Solar + Battery ---
            sol_r = self.solar.compute_hour(
                ghi_wm2        = ghi_profile[hour],
                dc_demand_kw   = dc_r["total_facility_power_kw"],
                soc_prev       = soc,
                ambient_temp_c = ambient_temps[hour]
            )
            soc = sol_r["soc_end"]

            # --- Energy balance: Solar thermal vs CCUS need ---
            Q_ccus_need     = mea_r["Q_solar_needed_kw"]    # kW still needed
            Q_solar_thermal = sol_r["Q_thermal_kw"]          # kW available

            # How much of CCUS solar need is met by solar thermal
            Q_solar_to_ccus  = min(Q_solar_thermal, Q_ccus_need)
            Q_thermal_excess = max(Q_solar_thermal - Q_ccus_need, 0)
            Q_ccus_deficit   = max(Q_ccus_need - Q_solar_thermal, 0)

            # Grid thermal backup covers any remaining deficit
            # (electric resistance heater or gas boiler — grid powered)
            Q_grid_thermal_kw = Q_ccus_deficit   # grid always fills the gap

            # Total CCUS heat coverage this hour (renewable fraction only)
            Q_total_covered   = mea_r["Q_waste_contribution_kw"] + Q_solar_to_ccus
            Q_total_required  = mea_r["Q_total_required_kw"]

            # Renewable coverage = what waste heat + solar covered (not grid backup)
            ccus_coverage_pct = (Q_total_covered / Q_total_required * 100) \
                                  if Q_total_required > 0 else 0

            # CCUS always runs at full capacity (grid thermal fills deficit)
            co2_captured_kgh  = self.mea.co2_available_kgh * self.mea.target_capture_rate

            # Grid thermal cost (at industrial gas/electric rate)
            GRID_THERMAL_COST = 0.06   # USD/kWh thermal
            grid_thermal_cost = Q_grid_thermal_kw * GRID_THERMAL_COST

            # Carbon credits this hour
            credits_usd = (co2_captured_kgh / 1000) * CARBON_CREDIT_USD_TONNE

            # Grid electricity cost this hour
            grid_cost_usd = sol_r["P_grid_import_kw"] * GRID_ELECTRICITY_USD_KWH

            # Combined renewable fraction (electricity + thermal)
            total_energy_in = (dc_r["total_facility_power_kw"] +
                               Q_total_required)
            renewable_in    = (sol_r["P_pv_kw"] + sol_r["discharge_kw"] +
                               mea_r["Q_waste_contribution_kw"] + Q_solar_to_ccus)
            combined_re_pct = min(renewable_in / total_energy_in * 100, 100) \
                              if total_energy_in > 0 else 0

            # --- Combine all results ---
            combined = {
                "hour"                   : hour,

                # DC
                "cpu_pct"                : dc_r["cpu_utilization_pct"],
                "dc_power_kw"            : dc_r["total_facility_power_kw"],
                "dc_waste_heat_kw"       : dc_r["q_total_waste_kw"],

                # HX
                "hx_Q_kw"                : hx_r["Q_actual_kw"],
                "glycol_out_c"           : hx_r["T_glycol_out_c"],

                # CCUS
                "Q_total_required_kw"    : round(Q_total_required, 2),
                "Q_waste_heat_kw"        : round(mea_r["Q_waste_contribution_kw"], 2),
                "Q_solar_to_ccus_kw"     : round(Q_solar_to_ccus, 2),
                "Q_ccus_deficit_kw"      : round(Q_ccus_deficit, 2),
                "ccus_coverage_pct"      : round(ccus_coverage_pct, 1),
                "co2_captured_kgh"       : round(co2_captured_kgh, 2),

                # Solar
                "ghi_wm2"                : ghi_profile[hour],
                "P_pv_kw"                : sol_r["P_pv_kw"],
                "Q_thermal_kw"           : sol_r["Q_thermal_kw"],
                "Q_thermal_excess_kw"    : round(Q_thermal_excess, 2),
                "soc_pct"                : round(soc * 100, 1),
                "P_grid_kw"              : sol_r["P_grid_import_kw"],

            # Economics
                "credits_usd"            : round(credits_usd, 4),
                "grid_cost_usd"          : round(grid_cost_usd + grid_thermal_cost, 4),
                "Q_grid_thermal_kw"      : round(Q_grid_thermal_kw, 2),
                "combined_re_pct"        : round(combined_re_pct, 1),
            }
            hourly_results.append(combined)

        # --- Daily Summary ---
        summary = self._summarize(hourly_results)

        return {"hourly": hourly_results, "summary": summary}

    # ------------------------------------------------------------------
    def _summarize(self, hourly: list) -> dict:
        """Aggregate 24 hourly results into daily KPIs."""

        total_co2_kg      = sum(r["co2_captured_kgh"]      for r in hourly)
        total_credits     = sum(r["credits_usd"]            for r in hourly)
        total_grid_cost   = sum(r["grid_cost_usd"]          for r in hourly)
        total_pv          = sum(r["P_pv_kw"]                for r in hourly)
        total_thermal     = sum(r["Q_thermal_kw"]           for r in hourly)
        total_waste_heat  = sum(r["Q_waste_heat_kw"]        for r in hourly)
        total_grid_import = sum(r["P_grid_kw"]              for r in hourly)
        total_q_required  = sum(r["Q_total_required_kw"]    for r in hourly)
        total_q_covered   = sum(r["Q_waste_heat_kw"] +
                                r["Q_solar_to_ccus_kw"]     for r in hourly)
        avg_re_pct        = np.mean([r["combined_re_pct"]   for r in hourly])
        avg_ccus_cov      = np.mean([r["ccus_coverage_pct"] for r in hourly])
        avg_cpu           = np.mean([r["cpu_pct"]           for r in hourly])

        # Annual extrapolation
        co2_annual_tonnes = total_co2_kg * 365 / 1000
        credits_annual    = total_credits * 365
        grid_cost_annual  = total_grid_cost * 365

        # Net benefit
        net_benefit_daily = total_credits - total_grid_cost

        return {
            # Daily totals
            "total_co2_captured_kg"     : round(total_co2_kg, 1),
            "total_carbon_credits_usd"  : round(total_credits, 2),
            "total_grid_cost_usd"       : round(total_grid_cost, 2),
            "net_benefit_usd_day"       : round(net_benefit_daily, 2),
            "total_pv_kwh"              : round(total_pv, 1),
            "total_solar_thermal_kwh"   : round(total_thermal, 1),
            "total_waste_heat_kwh"      : round(total_waste_heat, 1),
            "total_grid_import_kwh"     : round(total_grid_import, 1),
            "total_q_required_kwh"      : round(total_q_required, 1),
            "total_q_covered_kwh"       : round(total_q_covered, 1),

            # Averages
            "avg_cpu_pct"               : round(avg_cpu, 1),
            "avg_combined_re_pct"       : round(avg_re_pct, 1),
            "avg_ccus_coverage_pct"     : round(avg_ccus_cov, 1),

            # Annual projections
            "annual_co2_tonnes"         : round(co2_annual_tonnes, 1),
            "annual_credits_usd"        : round(credits_annual, 2),
            "annual_grid_cost_usd"      : round(grid_cost_annual, 2),
            "total_grid_thermal_kwh"    : round(sum(r.get("Q_grid_thermal_kw", 0) for r in hourly), 1),
        }

    # ------------------------------------------------------------------
    def optimize_sizing(
        self,
        cpu_profile:    list,
        ghi_profile:    list,
        target_ccus_re: float = 0.80,
        thermal_area_range: tuple = (100, 2000),
        step_m2:        float = 50.0
    ) -> dict:
        """
        Find minimum solar thermal area to achieve target CCUS
        renewable heat coverage.

        Sweeps thermal collector area from min to max in steps,
        runs full simulation at each size, returns optimal result.

        Parameters
        ----------
        target_ccus_re    : float   Target CCUS renewable coverage [0–1]. Default 0.80.
        thermal_area_range: tuple   (min_m2, max_m2) search range.
        step_m2           : float   Step size [m²].
        """

        print(f"\n  🔍 Optimizing solar thermal area for {target_ccus_re*100:.0f}% CCUS coverage...")
        print(f"     Sweeping {thermal_area_range[0]}–{thermal_area_range[1]} m² in {step_m2} m² steps")

        best_area   = None
        best_result = None
        sweep_data  = []

        areas = np.arange(thermal_area_range[0], thermal_area_range[1] + step_m2, step_m2)

        for area in areas:
            # Temporarily update solar model's thermal area
            original_area = self.solar.thermal_area_m2
            self.solar.thermal_area_m2 = area

            result = self.simulate_system(cpu_profile, ghi_profile)
            
            # Use daytime-only coverage (hours 6–18) — solar can't cover nights
            daytime_hours = [r for r in result["hourly"] if 6 <= r["hour"] <= 18]
            daytime_coverage = np.mean([r["ccus_coverage_pct"] for r in daytime_hours]) / 100.0
            coverage = daytime_coverage

            sweep_data.append({
                "thermal_area_m2"    : area,
                "ccus_coverage_pct"  : daytime_coverage * 100,
                "co2_captured_kg_day": result["summary"]["total_co2_captured_kg"],
                "credits_usd_day"    : result["summary"]["total_carbon_credits_usd"],
            })

            if coverage >= target_ccus_re and best_area is None:
                best_area   = area
                best_result = result

            # Restore
            self.solar.thermal_area_m2 = original_area

        if best_area is None:
            best_area   = thermal_area_range[1]
            self.solar.thermal_area_m2 = best_area
            best_result = self.simulate_system(cpu_profile, ghi_profile)
            self.solar.thermal_area_m2 = thermal_area_range[0]
            print(f"  ⚠️  Target not reached within range. Max area = {best_area} m²")
        else:
            print(f"  ✅ Optimal thermal area: {best_area:.0f} m² achieves "
                  f"{best_result['summary']['avg_ccus_coverage_pct']:.1f}% CCUS coverage")

        # Capital cost estimate for optimal area
        capex_thermal = best_area * SOLAR_THERMAL_CAPEX_M2
        capex_pv      = self.solar.pv_capacity_kwp * SOLAR_PV_CAPEX_KWP
        capex_battery = self.solar.battery_capacity_kwh * BATTERY_CAPEX_KWH
        capex_total   = capex_thermal + capex_pv + capex_battery

        # Simple payback
        annual_benefit = best_result["summary"]["annual_credits_usd"]
        payback_years  = capex_total / annual_benefit if annual_benefit > 0 else 999

        return {
            "optimal_thermal_area_m2" : best_area,
            "target_ccus_coverage_pct": target_ccus_re * 100,
            "achieved_coverage_pct"   : round(coverage * 100, 1),
            "capex_thermal_usd"       : round(capex_thermal, 0),
            "capex_pv_usd"            : round(capex_pv, 0),
            "capex_battery_usd"       : round(capex_battery, 0),
            "capex_total_usd"         : round(capex_total, 0),
            "annual_co2_tonnes"       : best_result["summary"]["annual_co2_tonnes"],
            "annual_credits_usd"      : round(best_result["summary"]["annual_credits_usd"], 2),
            "simple_payback_years"    : round(payback_years, 1),
            "sweep_data"              : sweep_data,
            "best_simulation"         : best_result,
        }


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 65)
    print("  SYSTEM OPTIMIZER — FULL PIPELINE TEST")
    print("=" * 65)

    # --- Build all components ---
    dc    = DataCenterThermalModel(it_capacity_kw=100, pue=1.4)
    hx    = HeatExchanger(area_m2=25.0, glycol_flow_kgs=0.8, glycol_inlet_temp_c=25.0)
    mea   = MEARegenerator(mea_flow_kgs=0.21, target_capture_rate=0.90)
    solar = SolarBatterySystem(
        pv_capacity_kwp      = 150.0,
        thermal_area_m2      = 300.0,
        battery_capacity_kwh = 200.0
    )

    # --- Fetch solar data ---
    print("\n  Loading solar data (NASA POWER or cached)...")
    cache_path = os.path.join("data", "nasa_solar_data.json")
    if os.path.exists(cache_path):
        import json
        with open(cache_path) as f:
            cached = json.load(f)
        ghi_annual = cached["ghi_wm2"]
        print(f"  ✅ Loaded from cache: {len(ghi_annual)} hourly values")
    else:
        ghi_annual = fetch_nasa_solar(lat=26.9, lon=73.0, year=2023)

    ghi_june = get_typical_day(ghi_annual, month=6)

    # --- Build optimizer ---
    opt = SystemOptimizer(dc, hx, mea, solar)

    # --- Run full simulation ---
    print("\n📊 Full System Simulation (300m² thermal, 150kWp PV, 200kWh battery)")
    print("─" * 75)

    cpu_profile = get_cpu_profile("office")
    result      = opt.simulate_system(cpu_profile, ghi_june)
    hourly      = result["hourly"]
    summary     = result["summary"]

    print(f"\n  {'Hr':>3} {'CPU%':>6} {'DC_kW':>7} {'WH_kW':>7} {'Q_sol':>7} "
          f"{'Q_def':>7} {'CCUS%':>7} {'CO2':>8} {'RE%':>6}")
    print(f"  {'─'*3} {'─'*6} {'─'*7} {'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*7} {'─'*8} {'─'*6}")

    for r in hourly:
        print(
            f"  {r['hour']:02d}:00"
            f"  {r['cpu_pct']:>5.1f}%"
            f"  {r['dc_power_kw']:>6.1f}kW"
            f"  {r['Q_waste_heat_kw']:>6.1f}kW"
            f"  {r['Q_solar_to_ccus_kw']:>6.1f}kW"
            f"  {r['Q_ccus_deficit_kw']:>6.1f}kW"
            f"  {r['ccus_coverage_pct']:>6.1f}%"
            f"  {r['co2_captured_kgh']:>7.2f}kg/h"
            f"  {r['combined_re_pct']:>5.1f}%"
        )

    print(f"\n  {'═'*65}")
    print(f"  DAILY SUMMARY")
    print(f"  {'─'*65}")
    print(f"  Avg CPU load               : {summary['avg_cpu_pct']:.1f}%")
    print(f"  Total CO₂ captured         : {summary['total_co2_captured_kg']:.1f} kg/day")
    print(f"  Annual CO₂ captured        : {summary['annual_co2_tonnes']:.1f} tonnes/year")
    print(f"  Total carbon credits       : ${summary['total_carbon_credits_usd']:.2f}/day")
    print(f"  Annual carbon credits      : ${summary['annual_credits_usd']:,.2f}/year")
    print(f"  Total grid cost            : ${summary['total_grid_cost_usd']:.2f}/day")
    print(f"  Net benefit                : ${summary['net_benefit_usd_day']:.2f}/day")
    print(f"  Avg CCUS heat coverage     : {summary['avg_ccus_coverage_pct']:.1f}%")
    print(f"  Avg combined RE fraction   : {summary['avg_combined_re_pct']:.1f}%")
    print(f"  Solar thermal (24h)        : {summary['total_solar_thermal_kwh']:.1f} kWh")
    print(f"  Waste heat used (24h)      : {summary['total_waste_heat_kwh']:.1f} kWh")
    print(f"  Grid import (24h)          : {summary['total_grid_import_kwh']:.1f} kWh")
    print(f"  Grid thermal backup (24h)  : {summary.get('total_grid_thermal_kwh', 0):.1f} kWh")

    # --- Run optimizer ---
    print(f"\n\n{'═'*65}")
    print(f"  SIZING OPTIMIZER — Target: 80% CCUS Renewable Coverage")
    print(f"{'═'*65}")

    opt_result = opt.optimize_sizing(
        cpu_profile    = cpu_profile,
        ghi_profile    = ghi_june,
        target_ccus_re = 0.80,
        thermal_area_range = (100, 1500),
        step_m2        = 50.0
    )

    print(f"\n  {'─'*55}")
    print(f"  OPTIMAL SYSTEM DESIGN")
    print(f"  {'─'*55}")
    print(f"  Solar thermal area needed  : {opt_result['optimal_thermal_area_m2']:.0f} m²")
    print(f"  CCUS coverage achieved     : {opt_result['achieved_coverage_pct']:.1f}%")
    print(f"  Annual CO₂ captured        : {opt_result['annual_co2_tonnes']:.1f} tonnes")
    print(f"  Annual carbon credits      : ${opt_result['annual_credits_usd']:,.2f}")
    print(f"  {'─'*55}")
    print(f"  CAPITAL COST BREAKDOWN")
    print(f"  {'─'*55}")
    print(f"  Solar thermal system       : ${opt_result['capex_thermal_usd']:>10,.0f}")
    print(f"  Solar PV (150 kWp)         : ${opt_result['capex_pv_usd']:>10,.0f}")
    print(f"  Battery storage (200 kWh)  : ${opt_result['capex_battery_usd']:>10,.0f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  TOTAL CAPEX                : ${opt_result['capex_total_usd']:>10,.0f}")
    print(f"  Simple payback period      : {opt_result['simple_payback_years']:.1f} years")

    print(f"\n  ✅ Module 5 complete — ready to build Streamlit dashboard")